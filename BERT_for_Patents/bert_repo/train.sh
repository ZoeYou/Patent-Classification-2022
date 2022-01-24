python run_classifier.py \
--task_name=PMLP \
--do_train=true \
--do_predict=true \
--data_dir=bert/PMLP/epo_data/epo_en_CLAIM1_from_2015 \
--vocab_file=../patent_model/Checkpoint/raw/bert_for_patents_vocab_39k.txt \
--bert_config_file=../patent_model/Checkpoint/raw/bert_for_patents_large_config.json \
--init_checkpoint=../patent_model/Checkpoint/raw/model.ckpt \
--label_file=./labels_group_id_4.tsv \
--output_dir=./bfp_en_claims_from2015_IPC4/

# bert for patent for english claims
