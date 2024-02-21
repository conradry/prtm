import logging
import hashlib
import os
import random
import tarfile
import time
import requests
from typing import Literal, List, Dict, Union
from tqdm import tqdm

from prtm.query.caching import cache_query

logger = logging.getLogger(__name__)
TQDM_BAR_FORMAT = '{l_bar}{bar}| {n_fmt}/{total_fmt} [elapsed: {elapsed} remaining: {remaining}]'
MMSEQS_PATH = os.path.expanduser(f"~/.prtm/mmseqs2/")


def api_call_wrapper(return_json: bool):
    """Wrapper for handling API calls with timeouts and retries"""
    def api_caller(func):
        def call_api(*args, **kwargs):
            while True:
                error_count = 0
                try:
                    result = func(*args, **kwargs)
                except requests.exceptions.Timeout:
                    logger.warning("Timeout while submitting to MSA server. Retrying...")
                    continue
                except Exception as e:
                    error_count += 1
                    logger.warning(f"Error while fetching result from MSA server. Retrying... ({error_count}/5)")
                    logger.warning(f"Error: {e}")
                    time.sleep(5)
                    if error_count > 5:
                        raise
                    continue
                break

            if return_json:
                try:
                    out = result.json()
                except ValueError:
                    logger.error(f"Server didn't reply with json: {result.text}")
                    out = {"status": "ERROR"}
                return out
            else:
                return None
    
        return call_api

    return api_caller
        

class MMSeqs2:
    def __init__(
        self,
        use_env: bool = True,
        use_filter: bool = True,
        use_templates: bool = False,
        use_pairing: bool = False,
        pairing_strategy: Literal["greedy", "complete"] = "greedy",
        host_url: str = "https://api.colabfold.com",
        user_agent: str = "",
    ):
        self.use_env = use_env
        self.use_filter = use_filter
        self.use_templates = use_templates
        self.use_pairing = use_pairing
        self.pairing_strategy = pairing_strategy
        self.host_url = host_url
        self.user_agent = user_agent
        if not self.user_agent:
            print("No user agent specified. Please set a user agent!")

        self.headers = {}
        self.submission_endpoint = "ticket/pair" if self.use_pairing else "ticket/msa"

        if self.user_agent:
            self.headers['User-Agent'] = user_agent

        if self.use_pairing:
            self.use_templates = False
            self.use_env = False
            if self.pairing_strategy == "greedy":
                self.mode = "pairgreedy"
            elif self.pairing_strategy == "complete":
                self.mode = "paircomplete"
        elif self.use_filter:
            self.mode = "env" if self.use_env else "all"
        else:
            self.mode = "env-nofilter" if self.use_env else "nofilter"

        self.N = 101  # an offset in the query

    @api_call_wrapper(return_json=True)
    def submit_request(
        self, sequences: List[str], mode: str, n: int = 101
    ) -> Dict[str, str]:
        query = ""
        for i, seq in enumerate(sequences):
            query += f">{self.N + i}\n{seq}\n"

        result = requests.post(
            f'{self.host_url}/{self.submission_endpoint}', 
            data={'q': query, 'mode': mode}, 
            timeout=6.02, 
            headers=self.headers,
        )
        
        return result

    @api_call_wrapper(return_json=True)
    def check_request_status(self, job_id: str):
        result = requests.get(
            f'{self.host_url}/ticket/{job_id}', timeout=6.02, headers=self.headers
        )
        return result

    @api_call_wrapper(return_json=False)
    def download_result(self, job_id: str, result_path: str):
        result = requests.get(
            f'{self.host_url}/result/download/{job_id}', timeout=6.02, headers=self.headers
        )
        with open(result_path, "wb") as out: 
            out.write(result.content)

        return result

    @api_call_wrapper(return_json=False)
    def request_templates(self, template_line: str, result_path: str):
        response = requests.get(
            f"{self.host_url}/template/{template_line}", stream=True, timeout=6.02, headers=self.headers
        )
        with tarfile.open(fileobj=response.raw, mode="r|gz") as tar:
            tar.extractall(path=result_path)

        os.symlink("pdb70_a3m.ffindex", f"{result_path}/pdb70_cs219.ffindex")
        with open(f"{result_path}/pdb70_cs219.ffdata", "w") as f:
            f.write("")

        return result_path

    @cache_query(
        hash_func_kwargs=["path_hash"],
        hash_class_attrs=[
            "use_env",
            "use_filter",
            "use_pairing",
            "pairing_strategy",
            "host_url",
        ],
    )
    def get_msas(
        self, unique_sequences: List[str], sequence_ids: List[int], path_hash: str
    ):
        path = os.path.join(MMSEQS_PATH, path_hash)
        os.makedirs(path, exist_ok=True)

        tar_gz_file = os.path.join(path, f"out.tar.gz")
        time_estimate = 150 * len(unique_sequences)
        with tqdm(total=time_estimate, bar_format=TQDM_BAR_FORMAT) as pbar:
            redo = True
            while redo:
                pbar.set_description("SUBMIT")
        
                # Resubmit job until it goes through
                out = self.submit_request(unique_sequences, self.mode, self.N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    time.sleep(sleep_time)
                    out = self.submit_request(unique_sequences, self.mode, self.N)
            
                if out["status"] == "ERROR":
                    raise Exception(f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence.')
            
                if out["status"] == "MAINTENANCE":
                    raise Exception(f'MMseqs2 API is undergoing maintenance. Please try again in a few minutes.')
            
                # wait for job to finish
                job_id = out["id"]
                cur_time = 0
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0,5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = self.check_request_status(job_id)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        cur_time += t
                        pbar.update(n=t)
            
                if out["status"] == "COMPLETE":
                    if cur_time < time_estimate:
                        pbar.update(n=(time_estimate - cur_time))
                    redo = False
            
                if out["status"] == "ERROR":
                    raise Exception(f'MMseqs2 API is giving errors. Please confirm your input is a valid protein sequence.')
        
            # Download results
            self.download_result(job_id, tar_gz_file)

        # prep list of a3m files
        if self.use_pairing:
            a3m_files = [os.path.join(path, "pair.a3m")]
        else:
            a3m_files = [os.path.join(path, "uniref.a3m")]
            if self.use_env: 
                a3m_files.append(os.path.join(path, "bfd.mgnify30.metaeuk30.smag30.a3m"))
                
        # extract a3m files
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

        # gather a3m lines
        a3m_lines = {}
        for a3m_file in a3m_files:
            update_M, M = True, None
            for line in open(a3m_file, "r"):
                if len(line) > 0:
                    if "\x00" in line:
                        line = line.replace("\x00", "")
                        update_M = True
                    if line.startswith(">") and update_M:
                        M = int(line[1:].rstrip())
                        update_M = False
                        if M not in a3m_lines: 
                            a3m_lines[M] = []   
                    a3m_lines[M].append(line)
                    
        return ["".join(a3m_lines[n]) for n in sequence_ids]

    @cache_query(
        hash_func_kwargs=["path_hash"],
        hash_class_attrs=[
            "use_templates",
            "host_url",
        ],
    )
    def get_templates(self, sequence_ids: List[int], path_hash: str):
        path = os.path.join(MMSEQS_PATH, path_hash)

        templates: Dict[int, List[str]] = {}
        for line in open(os.path.join(path, "pdb70.m8"), "r"):
            p = line.rstrip().split()
            M, pdb = p[0],p[1]
            M = int(M)
            if M not in templates: 
                templates[M] = []
            templates[M].append(pdb)

        template_paths = {}
        for k, template in templates.items():
            template_path = os.path.join(path, f"templates_{k}")
            os.makedirs(template_path, exist_ok=True)
            template_line = ",".join(template[:20])
            self.request_templates(template_line, template_path)
            template_paths[k] = template_path

        template_paths_ = []
        for n in sequence_ids:
            if n not in template_paths:
                template_paths_.append(None)
            else:
                template_paths_.append(template_paths[n])
                
        template_paths = template_paths_

        os.remove(os.path.join(path, "pdb70.m8"))

        return template_paths

    def query(self, sequences: Union[str, List[str]]):
        sequences = [sequences] if isinstance(sequences, str) else sequences
                
        # deduplicate and keep track of order
        unique_sequences = list(set(sequences))
        sequence_ids = [self.N + unique_sequences.index(seq) for seq in sequences]

        # Make a hash string of the sequences as a directory prefix
        hash_sha = hashlib.sha512()
        for seq in sequences:
            hash_sha.update(str(seq).encode("utf-8"))

        # A directory to store mmseqs2 outputs
        hash_str = hash_sha.hexdigest()[:16]
        path_hash = f"{hash_str}_{self.mode}"

        msas = self.get_msas(unique_sequences, sequence_ids, path_hash)
        output = {"msas" if not self.use_pairing else "paired_msas": msas}
        if self.use_templates:
            template_paths = self.get_templates(sequence_ids, path_hash)
            output["templates"] = template_paths

        # Cleanup unnecessary results files but don't delete any subdirectories
        for file in os.listdir(os.path.join(MMSEQS_PATH, path_hash)):
            path = os.path.join(MMSEQS_PATH, path_hash, file)
            if not os.path.isdir(path) and file != "pdb70.m8":
                os.remove(path)

        output["seq_ids"] = sequence_ids
        return output