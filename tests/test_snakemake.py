import subprocess

def test_run_snakemake_all():
    # subprocess.run(
    #     ['touch', 'test_input.txt']
    # )
    subprocess.run(
        ['snakemake', '-s', 'hypercluster.smk', '--config', 'input_data_files=test_input']
    )