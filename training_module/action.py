import asyncio
import functools


class ServerAction:
    LOCAL_TRAINING = "local_training"
    SEND_STATES = "send_states"
    GET_STATES = "get_states"

    def execute_action(self, action, worker_list):
        if action == self.LOCAL_TRAINING:
            self.local_training(worker_list)
        elif action == self.SEND_STATES:
            self.send_states(worker_list)
        elif action == self.GET_STATES:
            self.get_states(worker_list)

    @staticmethod
    def send_states(worker_list):
        # loop = asyncio.s
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        for worker in worker_list:
            task = asyncio.ensure_future(worker.send_config())
            tasks.append(task)
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

    @staticmethod
    def get_states(worker_list):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        for worker in worker_list:
            task = asyncio.ensure_future(worker.get_config())
            tasks.append(task)
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()

    @staticmethod
    def local_training(worker_list):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        tasks = []
        for worker in worker_list:
            task = asyncio.ensure_future(worker.local_training())
            tasks.append(task)
        loop.run_until_complete(asyncio.wait(tasks))
        loop.close()
