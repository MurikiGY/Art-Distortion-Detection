mkdir data
tar -xzf ./output.tgz -C data
unzip -q archive.zip -d data

cd data
mkdir original
mkdir modified

cd data

COUNT=0
for i in $(find images -type f)
do

    echo "$i"
    COUNT=$(($COUNT+1))
    mv "$i" original/ori_$COUNT.jpg
done

COUNT=0
for i in $(find output -type f)
do
    echo "$i"
    COUNT=$(($COUNT+1))
    mv "$i" modified/mod_$COUNT.jpg
done

rm -rf artists.csv resized images output
