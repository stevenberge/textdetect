cd rect
for file in `ls | grep png`; 
do tesseract $file tmp > /dev/null 2> /dev/null; 
  cat tmp.txt
done; 
rm *.png;
cd ..
