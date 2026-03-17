class BaseTaskDataset:

    def get_task_loaders(self, batch_size):
        """
        Returns:
            task1_loader
            task2_loader
        """
        raise NotImplementedError