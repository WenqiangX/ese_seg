set -v
python label_centerdeg.py
python cheby_fit.py
python train_coef_xml.py
python label_polygon.py
python val_polygon_xml.py
mkdir ../data/
ln -s ../sbd ../data/VOCsbdche
#rsync -av sbd_ESESEG_ImageSets/ ../data/VOCsbdche/ImageSets/Segmentation/
