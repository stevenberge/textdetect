scp -r *.cpp *.h *.hpp *.sh cmake sh py  x@desk:~/exp/
scp `find . | grep "[0-9][0-9][0-9].jpg$" | xargs` x@desk:~/exp/testset/

