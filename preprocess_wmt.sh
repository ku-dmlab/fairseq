
TEXT=examples/translation/wmt17_en_de

# run with all data to build a dictionary

for tp in train valid test; do
    for l in en de; do
        cat $TEXT/${tp}.$l >> $TEXT/${tp}_all.$l
        cat $TEXT/${tp}_iwslt.$l >> $TEXT/${tp}_all.$l
    done
done

fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train_all \
    --validpref $TEXT/valid_all \
    --testpref $TEXT/test_all \
    --destdir data-bin/wmt17.en-de.all \
    --workers 20

fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
    --destdir data-bin/wmt17.en-de \
    --workers 20 \
    --tgtdict data-bin/wmt17.en-de.all/dict.de.txt \
    --srcdict data-bin/wmt17.en-de.all/dict.en.txt

fairseq-preprocess --source-lang en --target-lang de \
    --trainpref $TEXT/train_iwslt --validpref $TEXT/valid_iwslt --testpref $TEXT/test_iwslt \
    --destdir data-bin/wmt17.en-de.iwslt \
    --workers 20 \
    --tgtdict data-bin/wmt17.en-de.all/dict.de.txt \
    --srcdict data-bin/wmt17.en-de.all/dict.en.txt
