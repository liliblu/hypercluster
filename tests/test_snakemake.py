import subprocess

def test_run_snakemake_all():
    subprocess.run(
        ['snakemake', '-s', 'hypercluster.smk', '--config', 'input_data_files=test_input']
    )