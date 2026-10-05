import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    from HDP_DS.scripts.run_workflow import run
    run()