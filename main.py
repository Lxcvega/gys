"""供应商选择系统入口：默认交互式 legacy，--hdp 运行 HDP-DS。"""
import argparse
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        description='供应商选择系统：默认运行 legacy；--hdp 后可传 HDP 参数。',
        epilog='HDP 参数帮助：python main.py --hdp --help',
        add_help=False,
        allow_abbrev=False,
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--hdp', action='store_true', help='运行 HDP-DS')
    modes.add_argument('--legacy', action='store_true', help='运行交互式传统流程（默认）')
    parser.add_argument('-h', '--help', action='store_true', help='显示帮助并退出')
    args, remaining = parser.parse_known_args(arguments)

    if args.hdp:
        # Forward every HDP option unchanged, including its own help flag.
        from HDP_DS.main import main as hdp_main
        forwarded = [arg for arg in arguments if arg != '--hdp']
        previous = sys.argv
        try:
            sys.argv = [previous[0], *forwarded]
            return hdp_main()
        finally:
            sys.argv = previous

    if args.help:
        parser.print_help()
        return
    if remaining:
        parser.error('legacy 不支持这些参数: ' + ' '.join(remaining))
    from legacy_main import main as legacy_main
    return legacy_main()


if __name__ == '__main__':
    main()
