
export PATH=$PATH:/c/Users/Hung/AppData/Local/Microsoft/WindowsApps/python3


for r in $(seq 0.1 0.1 0.8);
do
  python3 ex_unfair_2d.py --save --robust --radius $r
done
