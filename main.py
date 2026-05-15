import logging.config
import os

from configuration import config
from methods import METHODS

logging.config.fileConfig('configuration/logging.conf')
logger = logging.getLogger()


def setup_file_logging(args):
    log_dir = os.path.join(args.log_path, 'logs', args.dataset, args.note)
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, 'stdout.log')
    root_logger = logging.getLogger()

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and getattr(handler, 'baseFilename', None) == os.path.abspath(log_file):
            return log_file

    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d > %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    root_logger.addHandler(file_handler)
    return log_file


def main():
    # Get Configurations
    args = config.base_parser()
    log_file = setup_file_logging(args)
    logger.info('Mirroring logs to: %s', log_file)
    logger.info('Running for seeds: %s', args.seeds)
    for seed in args.seeds:
        setattr(args, 'rnd_seed', seed)
        logger.info('Configuration: %s', args)

        trainer = METHODS[args.method](**vars(args))
        trainer.run()

if __name__ == "__main__":
    main()
