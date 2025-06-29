import argparse

from openlrm.runners import REGISTRY_RUNNERS


def main():

    parser = argparse.ArgumentParser(description='OpenLRM launcher')
    parser.add_argument('runner', type=str, help='Runner to launch')
    args, unknown = parser.parse_known_args()

    if args.runner not in REGISTRY_RUNNERS:
        raise ValueError('Runner {} not found'.format(args.runner))

    RunnerClass = REGISTRY_RUNNERS[args.runner]
    with RunnerClass() as runner:
        runner.run()


if __name__ == '__main__':
    main()
