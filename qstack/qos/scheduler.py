from typing import Any, Dict, List
from threading import Thread, Lock, Semaphore

from qstack.qernel.qernel import Qernel, QernelArgs
from qstack.types import QOSEngineI, scheduler_policy, Job, Scheduler_base


class Scheduler(Scheduler_base):
	'''Local scheduler'''

	def __init__(self):
		pass

	def register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
		new_thread = Thread(target=self.__register_qernel, args=(self, qernel, compile_args))
		pass

	def __register_qernel(self, qernel: Qernel, compile_args: Dict[str, Any]) -> int:
		self.queue_lock.acquire()
		#For now the advise method is actually changing the queue itself instead of
		# proposing where the next job should go, this is way I left the acquire and
		# release here, just because the changing of the queue might should be outside of
		# the advise method 
		self.policy.advise()
		self.queue_lock.release()

	def __execute_qernel(self, qid: int, args: QernelArgs, shots: int) -> None:
		
		#When should this thread be killed?
		while(1):
			self.queue_counter.acquire() #If the queue_counter semaphore is higher than 0 there is job/s to run
			self.queue_lock.acquire()
			next_job = self.queue.pop(0)
			self.queue_lock.release()
			#TODO Execute next_job

class fifo_policy(scheduler_policy):
	'''First Come First Served Policy or First In First Out'''
	def advise(self, new_job:Job):
		super.queue.append(new_job)
		super.queue_counter.release() #Adding one to the queue_counter semaphore