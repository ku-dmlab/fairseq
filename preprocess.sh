
pip install --editable ./
cd examples/translation/
bash prepare-iwslt14-pe.sh
cd ../..

TEXT=examples/translation/iwslt14.tokenized.de-en

# run with all data to build a dictionary

for tp in train valid test; do
    for l in en de; do
        cat $TEXT/${tp}.$l >> $TEXT/${tp}_all.$l
        cat $TEXT/${tp}_mt.$l >> $TEXT/${tp}_all.$l
        cat $TEXT/${tp}_pe.$l >> $TEXT/${tp}_all.$l
    done
done

fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train_all \
    --validpref $TEXT/valid_all \
    --testpref $TEXT/test_all \
    --destdir data-bin/iwslt14.tokenized.en-de.all \
    --workers 20

fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/iwslt14.tokenized.en-de \
    --workers 20 \
    --tgtdict data-bin/iwslt14.tokenized.en-de.all/dict.de.txt \
    --srcdict data-bin/iwslt14.tokenized.en-de.all/dict.en.txt

fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train_mt --validpref $TEXT/valid_mt --testpref $TEXT/test_mt \
    --destdir data-bin/iwslt14.tokenized.en-de.mt \
    --workers 20 \
    --tgtdict data-bin/iwslt14.tokenized.en-de.all/dict.de.txt \
    --srcdict data-bin/iwslt14.tokenized.en-de.all/dict.en.txt

fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train_pe --validpref $TEXT/valid_pe --testpref $TEXT/test_pe \
    --destdir data-bin/iwslt14.tokenized.en-de.pe \
    --workers 20 \
    --tgtdict data-bin/iwslt14.tokenized.en-de.all/dict.de.txt \
    --srcdict data-bin/iwslt14.tokenized.en-de.all/dict.en.txt
