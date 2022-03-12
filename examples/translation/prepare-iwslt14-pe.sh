#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=subword-nmt/subword_nmt
BPE_TOKENS=10000

URL="http://dl.fbaipublicfiles.com/fairseq/data/iwslt14/de-en.tgz"
GZ=de-en.tgz

URL_PE="https://docs.google.com/uc?export=download&id=1BShX8FsKAFGHPQ0vXrk_VjqRY_-6Y3K2"
PED=pe

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=de
tgt=en
lang=de-en
prep=iwslt14.tokenized.de-en
tmp=$prep/tmp
orig=orig

mkdir -p $orig $tmp $prep

echo "Downloading data from ${URL}..."
cd $orig
wget "$URL"

if [ -f $GZ ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

tar zxvf $GZ

echo "Downloading data from ${URL_PE}..."
wget --no-check-certificate "$URL_PE" -O $PED.zip

if [ -f $PED.zip ]; then
    echo "Data successfully downloaded."
else
    echo "Data not successfully downloaded."
    exit
fi

unzip $PED.zip -d $PED

cd ..

echo "pre-processing train data..."
for l in $src $tgt; do
    f=train.tags.$lang.$l
    tok=train.tags.$lang.tok.$l

    cat $orig/$lang/$f | \
    grep -v '<url>' | \
    grep -v '<talkid>' | \
    grep -v '<keywords>' | \
    sed -e 's/<title>//g' | \
    sed -e 's/<\/title>//g' | \
    sed -e 's/<description>//g' | \
    sed -e 's/<\/description>//g' | \
    perl $TOKENIZER -threads 8 -l $l > $tmp/$tok
    echo ""
done
perl $CLEAN -ratio 1.5 $tmp/train.tags.$lang.tok $src $tgt $tmp/train.tags.$lang.clean 1 175
for l in $src $tgt; do
    perl $LC < $tmp/train.tags.$lang.clean.$l > $tmp/train.tags.$lang.$l
done

echo "pre-processing valid/test data..."
for l in $src $tgt; do
    for o in `ls $orig/$lang/IWSLT14.TED*.$l.xml`; do
    fname=${o##*/}
    f=$tmp/${fname%.*}
    echo $o $f
    grep '<seg id' $o | \
        sed -e 's/<seg id="[0-9]*">\s*//g' | \
        sed -e 's/\s*<\/seg>\s*//g' | \
        sed -e "s/\’/\'/g" | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $f
    echo ""
    done
done

echo "pre-processing post-edit data..."
cat $orig/$PED/Train/train.src | perl $TOKENIZER -threads 8 -l en | perl $LC > $tmp/tv_mt.en
cp $tmp/tv_mt.en $tmp/tv_pe.en
cat $orig/$PED/Train/train.mt | perl $TOKENIZER -threads 8 -l de | perl $LC > $tmp/tv_mt.de
cat $orig/$PED/Train/train.pe | perl $TOKENIZER -threads 8 -l de | perl $LC > $tmp/tv_pe.de
cat $orig/$PED/Dev/dev.src | perl $TOKENIZER -threads 8 -l en | perl $LC > $tmp/test_mt.en
cp $tmp/test_mt.en $tmp/test_pe.en
cat $orig/$PED/Dev/dev.mt | perl $TOKENIZER -threads 8 -l de | perl $LC > $tmp/test_mt.de
cat $orig/$PED/Dev/dev.pe | perl $TOKENIZER -threads 8 -l de | perl $LC > $tmp/test_pe.de

echo "creating mt/pe train, valid, test..."
for typ in mt pe; do
    for l in $src $tgt; do
        awk '{if (NR%7 == 0)  print $0; }' $tmp/tv_$typ.$l > $tmp/valid_$typ.$l
        awk '{if (NR%7 != 0)  print $0; }' $tmp/tv_$typ.$l > $tmp/train_$typ.$l
    done
done

echo "creating train, valid, test..."
for l in $src $tgt; do
    awk '{if (NR%23 == 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/valid.$l
    awk '{if (NR%23 != 0)  print $0; }' $tmp/train.tags.de-en.$l > $tmp/train.$l

    cat $tmp/IWSLT14.TED.dev2010.de-en.$l \
        $tmp/IWSLT14.TEDX.dev2012.de-en.$l \
        $tmp/IWSLT14.TED.tst2010.de-en.$l \
        $tmp/IWSLT14.TED.tst2011.de-en.$l \
        $tmp/IWSLT14.TED.tst2012.de-en.$l \
        > $tmp/test.$l
done

TRAIN=$tmp/train.en-de
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
    cat $tmp/train_mt.$l >> $TRAIN
    cat $tmp/train_pe.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
    for typ in mt pe; do
        for f in train_$typ.$L valid_$typ.$L test_$typ.$L; do
            echo "apply_bpe.py to ${f}..."
            python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
        done
    done 
done
