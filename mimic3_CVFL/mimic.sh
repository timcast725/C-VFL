python -m mimic3benchmark.scripts.extract_subjects ../physionet.org/files/mimiciii/1.4 data/root/
python -m mimic3benchmark.scripts.validate_events data/root/
python -m mimic3benchmark.scripts.extract_episodes_from_subjects data/root/
python -m mimic3benchmark.scripts.split_train_and_test data/root/
python -m mimic3benchmark.scripts.create_in_hospital_mortality data/root/ data/in-hospital-mortality/
python -m mimic3benchmark.scripts.create_decompensation data/root/ data/decompensation/
python -m mimic3benchmark.scripts.create_length_of_stay data/root/ data/length-of-stay/
python -m mimic3benchmark.scripts.create_phenotyping data/root/ data/phenotyping/
python -m mimic3benchmark.scripts.create_multitask data/root/ data/multitask/

declare -a StringArray=("data/in-hospital-mortality/" "data/decompensation/" "data/length-of-stay/" "data/phenotyping/" "data/multitask/" )
for val in "${StringArray[@]}"; do
    python -m mimic3models.split_train_val $val
done
