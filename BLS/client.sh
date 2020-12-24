nohup python3 -u client2.py > test.txt 2>&1 &
nohup python3 -u client1.py > test.txt 2>&1 &  
nohup python3 -u client3.py > test.txt 2>&1 & 
