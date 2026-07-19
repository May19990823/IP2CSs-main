import multiprocessing
import argparse
import json
import os
import signal
from functools import partial
from multiprocessing import Manager
from pathlib import Path
from time import time
import ase.io
import gurobipy as gb
import pandas as pd
from tqdm import tqdm

from Utils.compositional import generate_stoichiometric_info
from Utils.config_manager import Config
from ip4ch.IntegerProgram import Allocate
from ip4ch.generate_structure_entry import read_json_entries, process_entry
from ipcss.LatticeGridEngine import LatticeGridEngine


def _optional_config_value(config, *keys):
    current = getattr(config, "resolved_config", config)
    for key in keys:
        if not hasattr(current, "get"):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def resolve_mlip_pair_paths(config):
    pair_dir = _optional_config_value(config, "MLIP", "pair_dir")
    if pair_dir:
        return None, str(pair_dir)

    pair_file = _optional_config_value(config, "MLIP", "pair_file")
    if pair_file:
        return str(pair_file), None

    raise ValueError("MLIP must define pair_dir or pair_file")


# 修改1：使用Manager创建共享对象
class SharedResources:
    _instance = None

    def __init__(self):
        self.manager = Manager()
        self.stop_event = self.manager.Event()
        self.lock = self.manager.Lock()

    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance


def signal_handler(sig, frame):
    """处理Ctrl+C终止信号"""
    shared = SharedResources.get_instance()
    shared.stop_event.set()  # 修改为使用Event
    print("\n[Main] Termination signal received. Waiting for workers to finish...")


def _worker_initializer(shared_resources):
    """子进程初始化（Windows必须）"""
    global _worker_shared
    _worker_shared = shared_resources


class ParallelSolver:

    def __init__(self, config_path):
        self.config = self._load_config(config_path)
        self._init_paths()
        self._prepare_gurobi_env()
        signal.signal(signal.SIGINT, signal_handler)

        # 初始化共享资源
        self.shared = SharedResources.get_instance()

    def _load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        return Config(config_path=config_path)

    def _init_paths(self):
        """初始化所有路径"""
        self.program_root = self.config.get("base", "root")
        self.data_dir = self.config.get("base", "data")
        self.output_dir = self.config.get("base", "output")
        self.grids_dir = self.config.get("input", "grids")
        self.effective_pair_dir = self.config.get("input", "effective_pair")
        self.entry_dir = self.config.get("input", "entry")
        self.ip_results_dir = self.config.get("output", "ip_results")
        self.plot_results_dir = self.config.get("output", "plot_results")
        os.makedirs(self.output_dir, exist_ok=True)

    def _prepare_gurobi_env(self):
        """配置Gurobi环境（集群版需要）"""
        self.gurobi_params = {
            "PoolSolutions": self.config.get("Gurobi", "PoolSolutions"),
            "PoolSearchMode": self.config.get("Gurobi", "PoolSearchMode"),
            "TimeLimit": self.config.get("Gurobi", "TimeLimit"),
            "ThreadsPerProcess": self.config.get("Gurobi", "Threads"),  # 每个进程的线程数
            "MIPGap": self.config.get("Gurobi", "MIPGap"),
            "MIPFocus": self.config.get("Gurobi", "MIPFocus"),
            "Seed": self.config.get("Gurobi", "Seed"),
            "Backend": self.config.get("Gurobi", "Backend"),
            "WriteModel": bool(self.config.get("Gurobi", "WriteModel")),
            "NodefileDir": os.environ.get("GUROBI_NODEFILE_DIR", "/tmp/gurobi_nodes")
        }
        os.makedirs(self.gurobi_params["NodefileDir"], exist_ok=True)

    @staticmethod
    def _available_cpu_count():
        try:
            return len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            pass

        for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
            value = os.environ.get(env_name)
            if value:
                try:
                    return int(value)
                except ValueError:
                    pass

        return multiprocessing.cpu_count()

    @staticmethod
    def _unpack_optimization_result(result):
        structures, runtime, best_obj, pool_objectives, selected_vars, is_optimal, status_code = result[:7]
        solver_summary = result[7] if len(result) > 7 else {}
        solution_records = result[8] if len(result) > 8 else []
        skipped_solutions = result[9] if len(result) > 9 else []
        return {
            "structures": structures,
            "runtime": runtime,
            "best_obj": best_obj,
            "pool_objectives": pool_objectives,
            "selected_vars": selected_vars,
            "is_optimal": is_optimal,
            "status_code": status_code,
            "solver_summary": solver_summary,
            "solution_records": solution_records,
            "skipped_solutions": skipped_solutions,
        }

    @staticmethod
    def _json_safe(value):
        if isinstance(value, dict):
            return {str(key): ParallelSolver._json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ParallelSolver._json_safe(item) for item in value]
        try:
            import numpy as np
            if isinstance(value, np.generic):
                return value.item()
        except Exception:
            pass
        return value

    @staticmethod
    def _append_jsonl(path, payload):
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(ParallelSolver._json_safe(payload), ensure_ascii=False) + "\n")

    @staticmethod
    def _append_ip_obj_row(output_dir, entry_id, unpacked, written_count=None):
        ip_obj_path = os.path.join(output_dir, "IP_OBJ.txt")
        solver_summary = unpacked.get("solver_summary", {})
        if written_count is None:
            written_count = len(unpacked.get("structures", []))
        need_header = not os.path.exists(ip_obj_path) or os.path.getsize(ip_obj_path) == 0
        with open(ip_obj_path, "a", encoding="utf-8") as f:
            if need_header:
                f.write(
                    "EntryID\tStatus\tBestObj\tObjBound\tMIPGap\t"
                    "SolveTime_s\tWrittenSolutions\n"
                )
            f.write(
                f"{entry_id}\t"
                f"{solver_summary.get('gurobi_status_name')}\t"
                f"{'' if unpacked.get('best_obj') is None else format(unpacked.get('best_obj'), '.8f')}\t"
                f"{'' if solver_summary.get('objective_bound') is None else format(solver_summary.get('objective_bound'), '.8f')}\t"
                f"{'' if solver_summary.get('mip_gap') is None else format(solver_summary.get('mip_gap'), '.8g')}\t"
                f"{'' if unpacked.get('runtime') is None else format(unpacked.get('runtime'), '.3f')}\t"
                f"{int(written_count)}\n"
            )

    @staticmethod
    def _solver_status_payload(
            entry_id,
            specific_phase_name,
            unpacked,
            solution_records,
            written_structure_files,
            status_override=None,
            error=None):
        solver_summary = unpacked.get("solver_summary", {})
        payload = {
            "EntryID": entry_id,
            "SpecificPhaseName": specific_phase_name,
            "Status": status_override or solver_summary.get("gurobi_status_name"),
            "BestObj": unpacked.get("best_obj"),
            "SolveTime_s": unpacked.get("runtime"),
            "IsOptimal": unpacked.get("is_optimal"),
            "GurobiStatusCode": unpacked.get("status_code"),
            "GurobiStatusName": solver_summary.get("gurobi_status_name"),
            "HasIncumbent": solver_summary.get("has_incumbent"),
            "ObjectiveBound": solver_summary.get("objective_bound"),
            "MIPGap": solver_summary.get("mip_gap"),
            "RawSolutionCount": solver_summary.get("solution_count_raw"),
            "WrittenSolutionCount": len(written_structure_files),
            "StructureFiles": list(written_structure_files),
            "SkippedSolutionCount": len(unpacked.get("skipped_solutions", [])),
            "SolutionRecords": solution_records,
            "SkippedSolutions": unpacked.get("skipped_solutions", []),
        }
        if error is not None:
            payload["Error"] = str(error)
        return payload

    def _generate_tasks(self):
        """生成所有待处理任务"""
        chemical_space = sorted(self.config.get("Elements"))
        phase = ''.join(sorted(chemical_space))
        self.phase = phase
        self.test_id = str(self.config.get("Test_id"))

        stoichiometric_list, stoichiometric_dict_list = generate_stoichiometric_info(
            species_list=chemical_space,
            stoichiometric_range_list=self.config.get("StoichiometricRatio"),
            specific_stoichiometric_list=self.config.get("SpecificStoichiometric")
        )

        # Generate solver-ready entries with the local IPCSs LatticeGridEngine.
        # The engine writes one JSON per expanded formula, e.g. B1 * 10 -> B10.json.
        engine = LatticeGridEngine(self.config.resolved_config)
        pair_file, pair_dir = resolve_mlip_pair_paths(self.config)
        entry_summaries = engine.generate_entries(
            EntryDir=self.entry_dir,
            name=self.phase,
            stoichiometric_dict_list=stoichiometric_dict_list,
            multiplicities=self.config.get("Multiplicities"),
            dry_run=False,
        )

        "返回的总task列表，[[组分1task列表],[组分2task列表],...]"
        total_task_list = []
        for entry_summary in entry_summaries:
            specific_phase_name = entry_summary["FullFormula"]
            specific_task = []
            entry_file = os.path.join(self.entry_dir, self.phase, f'{specific_phase_name}.json')
            if not os.path.exists(entry_file):
                raise FileNotFoundError(entry_file)

            for entry in read_json_entries(entry_file):
                specific_task.append({
                    "program_root": self.program_root,
                    "data_dir": self.data_dir,
                    "output_dir": self.output_dir,
                    "entry_dir": self.entry_dir,
                    "grids_dir": self.grids_dir,
                    "effective_pair_dir": self.effective_pair_dir,
                    "ip_results_dir": self.ip_results_dir,
                    "plot_dir": self.plot_results_dir,
                    "entry": entry,
                    "phase": phase,
                    "r_min": self.config.get("MLIP", "r_min"),
                    "r_max": self.config.get("MLIP", "r_max"),
                    "pair_type": str(self.config.get("MLIP", "pair_type")),
                    "pair_file": pair_file,
                    "pair_dir": pair_dir,
                    "test_id": self.test_id,
                    "specific formula": specific_phase_name,
                    "compound type": str(self.config.get("Compound"))
                })
            if specific_task:
                total_task_list.append(specific_task)
            else:
                print(f"[Main] No solver entries generated for {specific_phase_name}; skipping")
        return total_task_list

    @staticmethod
    def _worker_process(specific_task, gurobi_params):
        """工作进程核心逻辑（移除stop_flag参数）"""
        shared = SharedResources.get_instance()
        if shared.stop_event.is_set():
            return None

        try:
            # 每个进程独立Gurobi环境
            # with gb.Env(params=gurobi_params) as env, gb.Model(env=env) as m:

            structure = process_entry(specific_task["entry"])  # 新增结构处理
            allocation = Allocate(
                ProgramRoot=specific_task['program_root'],
                DataDir=specific_task["data_dir"],
                OutputDir=specific_task["output_dir"],
                EntryDir=specific_task["entry_dir"],
                GridsDir=specific_task["grids_dir"],
                EffectivePairDir=specific_task["effective_pair_dir"],
                IPResultsDir=specific_task["ip_results_dir"],
                PlotDir=specific_task["plot_dir"],
                lattice=specific_task["entry"]["Lattice"],
                lattice_matrix=structure.lattice.matrix,  # 修复缺失的lattice_matrix
                lattice_type=specific_task["entry"]["Lattice Type"],
                grid_size=specific_task["entry"]["Grid Size"],
                full_formula_dict=specific_task["entry"]["Full Formula Dictionary"],
                r_min=specific_task["r_min"],
                r_max=specific_task["r_max"],
                compound_type=specific_task["compound type"],
                pair_type=specific_task["pair_type"],
                pair_file=specific_task["pair_file"],
                pair_dir=specific_task["pair_dir"],
            )

            """result = allocation.optimize_symmetry_ase(
                group=specific_task["entry"]["Space Group"],
                PoolSolutions=gurobi_params["PoolSolutions"],
                TimeLimit=gurobi_params["TimeLimit"],
                threads=gurobi_params["ThreadsPerProcess"]
            )
            if result:
                # 结果处理
                return {
                    "status": "success",
                    "entry_id": specific_task["entry"]["Index"],
                    "specific_phase_name": specific_task["specific formula"],
                    "result": result,
                    "energy": result[3] if result else None
                }"""

            model_file = None
            if gurobi_params.get("WriteModel", False):
                model_dir = os.path.join(
                    specific_task["ip_results_dir"],
                    specific_task["phase"],
                    specific_task["specific formula"],
                    str(specific_task["test_id"]),
                    "models",
                )
                model_file = os.path.join(model_dir, f"model_entry_{specific_task['entry']['Index']}.lp")

            result = allocation.optimize_symmetry_ase(
                group=specific_task["entry"]["Space Group"],
                PoolSolutions=gurobi_params["PoolSolutions"],
                TimeLimit=gurobi_params["TimeLimit"],
                threads=gurobi_params["ThreadsPerProcess"],
                backend=gurobi_params["Backend"],
                mip_gap=gurobi_params["MIPGap"],
                mip_focus=gurobi_params["MIPFocus"],
                seed=gurobi_params["Seed"],
                pool_search_mode=gurobi_params["PoolSearchMode"],
                write_model=gurobi_params.get("WriteModel", False),
                model_file=model_file,
            )
            if result is not None:
                unpacked = ParallelSolver._unpack_optimization_result(result)
                solver_summary = unpacked["solver_summary"]
                return {
                    "status": "success",
                    "entry_id": specific_task["entry"]["Index"],
                    "specific_phase_name": specific_task["specific formula"],
                    "result": result,
                    "best_obj": unpacked["best_obj"],
                    "runtime": unpacked["runtime"],
                    "is_optimal": unpacked["is_optimal"],
                    "status_code": unpacked["status_code"],
                    "solver_summary": solver_summary,
                    "solution_records": unpacked["solution_records"],
                    "skipped_solutions": unpacked["skipped_solutions"],
                    "solution_count_written": len(unpacked["structures"]),
                }

        except gb.GurobiError as e:
            return {
                "status": "gurobi_error",
                "entry_id": specific_task.get("entry", {}).get("Index"),
                "specific_phase_name": specific_task.get("specific formula"),
                "error": str(e),
            }
        except Exception as e:
            return {
                "status": "error",
                "entry_id": specific_task.get("entry", {}).get("Index"),
                "specific_phase_name": specific_task.get("specific formula"),
                "error": str(e),
            }

    def run(self):

        """启动并行计算"""
        total_tasks_list = self._generate_tasks()
        # print(f"[Main] Total tasks: {len(tasks)}")

        "按照不同组分分别处理specific task"
        specific_phase_name = ''
        for specific_task in total_tasks_list:
            global_minimum_found = 0

            # 确定该specific task的化学式
            specific_phase_name = specific_task[0]["specific formula"]
            # 资源分配策略
            available_cpus = self._available_cpu_count()
            threads_per_worker = int(self.gurobi_params["ThreadsPerProcess"])
            total_workers = min(
                max(1, available_cpus // threads_per_worker),
                len(specific_task))
            """total_workers = min(
                    18 // self.gurobi_params["ThreadsPerProcess"],
                    len(tasks))"""
            print(
                f"[Main] Using {total_workers} workers for searching {specific_phase_name}, "
                f"each with {threads_per_worker} threads "
                f"(available_cpus={available_cpus})")

            # 修改2：正确配置进程池
            start_time = time()
            with multiprocessing.Pool(
                    processes=total_workers,
                    initializer=_worker_initializer,
                    initargs=(self.shared,)
            ) as pool:
                worker = partial(self._worker_process, gurobi_params=self.gurobi_params)

                # 异步提交任务
                try:
                    results = pool.imap_unordered(worker, specific_task, chunksize=1)

                    # 进度监控
                    with tqdm(total=len(specific_task), desc="Processing Entries") as pbar:
                        for res in results:
                            successful_flag = self._handle_result(res, specific_phase_name)
                            if successful_flag:
                                global_minimum_found += successful_flag
                            pbar.update(1)

                            # 检查停止标志
                            if self.shared.stop_event.is_set():
                                pool.terminate()
                                break
                except KeyboardInterrupt:
                    self.shared.stop_event.set()
                    pool.terminate()

            end_time = time()

            log_data = {
                "Specific Phase Name": specific_phase_name,
                "Z": self.config.get("Multiplicities"),
                "PoolSolutions": self.gurobi_params["PoolSolutions"],
                "Total Entries": len(specific_task),
                "ThreadsPerProcess": self.gurobi_params["ThreadsPerProcess"],
                "Total Workers": total_workers,
                "Successful Solved Entries": global_minimum_found,
                "Solving Time(h)": (end_time - start_time)/3600,

            }
            log_df = pd.DataFrame([log_data])
            # 保存 CSV
            log_csv_path = os.path.join(self.ip_results_dir, self.phase, specific_phase_name, self.test_id,
                                        "log_info.csv")
            os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
            log_df.to_csv(log_csv_path, index=False, encoding='utf-8')
            print("[Main] All tasks completed")

    def _handle_result(self, result, specific_phase_name):
        successful_flag = 0
        """处理单个结果"""
        if not result:
            return 0

        output_dir = os.path.join(
            self.ip_results_dir,
            self.phase,
            specific_phase_name,
            self.test_id,
        )

        if result["status"] != "success":
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                solver_status_path = os.path.join(output_dir, "solver_status.jsonl")
                self._append_jsonl(solver_status_path, {
                    "EntryID": result.get("entry_id"),
                    "SpecificPhaseName": specific_phase_name,
                    "Status": result.get("status"),
                    "Error": result.get("error", "Unknown error"),
                    "WrittenSolutionCount": 0,
                })
            print(f"[Error] Task failed: {result.get('error', 'Unknown error')}")
            return 0

        os.makedirs(output_dir, exist_ok=True)
        solver_status_path = os.path.join(output_dir, "solver_status.jsonl")

        try:
            unpacked = self._unpack_optimization_result(result["result"])
            structure_data = unpacked["structures"]
        except (IndexError, TypeError, ValueError) as e:
            self._append_jsonl(solver_status_path, {
                "EntryID": result.get("entry_id"),
                "SpecificPhaseName": specific_phase_name,
                "Status": "malformed_result",
                "Error": str(e),
                "WrittenSolutionCount": 0,
            })
            print(f"[Error] Malformed result data: {str(e)}")
            return 0

        solution_records = [dict(record) for record in unpacked.get("solution_records", [])]
        written_structure_files = []

        if len(structure_data) == 0:
            self._append_jsonl(
                solver_status_path,
                self._solver_status_payload(
                    entry_id=result["entry_id"],
                    specific_phase_name=specific_phase_name,
                    unpacked=unpacked,
                    solution_records=solution_records,
                    written_structure_files=written_structure_files,
                ),
            )
            self._append_ip_obj_row(output_dir, result["entry_id"], unpacked, written_count=0)
            print(
                f"[Main] Entry {result['entry_id']} finished with "
                f"status={unpacked['solver_summary'].get('gurobi_status_name')} "
                "and no written structures"
            )
            return 0

        try:
            if len(structure_data) > 1:
                for structure_index, atoms in enumerate(structure_data):
                    structure_filename = os.path.join(
                        output_dir,
                        f"struct_{result['entry_id']}_{structure_index}.vasp",
                    )
                    ase.io.write(
                        structure_filename,
                        images=atoms,
                        format='vasp'
                    )
                    written_structure_files.append(structure_filename)
                    if structure_index < len(solution_records):
                        solution_records[structure_index]["structure_file"] = structure_filename
                    print(f"optimal结构已经保存到{structure_filename}")
            else:
                structure_filename = os.path.join(output_dir, f"struct_{result['entry_id']}.vasp")
                ase.io.write(
                    structure_filename,
                    images=structure_data[0],
                    format='vasp'
                )
                written_structure_files.append(structure_filename)
                if solution_records:
                    solution_records[0]["structure_file"] = structure_filename
                print(f"optimal结构已经保存到{structure_filename}")
        except Exception as e:
            self._append_jsonl(
                solver_status_path,
                self._solver_status_payload(
                    entry_id=result["entry_id"],
                    specific_phase_name=specific_phase_name,
                    unpacked=unpacked,
                    solution_records=solution_records,
                    written_structure_files=written_structure_files,
                    status_override="write_error",
                    error=e,
                ),
            )
            self._append_ip_obj_row(
                output_dir,
                result["entry_id"],
                unpacked,
                written_count=len(written_structure_files),
            )
            print(f"[Error] Failed to write structures for entry {result['entry_id']}: {str(e)}")
            return 0

        self._append_jsonl(
            solver_status_path,
            self._solver_status_payload(
                entry_id=result["entry_id"],
                specific_phase_name=specific_phase_name,
                unpacked=unpacked,
                solution_records=solution_records,
                written_structure_files=written_structure_files,
            ),
        )
        self._append_ip_obj_row(
            output_dir,
            result["entry_id"],
            unpacked,
            written_count=len(written_structure_files),
        )
        successful_flag = 1

        return successful_flag


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the Boron lattice/IP search")
    parser.add_argument(
        "--config",
        default=os.path.join(Path(__file__).parent, "config.yml"),
        help="Path to the run-specific YAML configuration",
    )
    return parser.parse_args()


if __name__ == '__main__':
    # Windows下必须的初始化
    _ = SharedResources.get_instance()
    solver = ParallelSolver(_parse_args().config)
    solver.run()
