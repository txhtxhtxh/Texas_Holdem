- Expected value calculating:


在linux下（其他系统不支持计算EV）先编译生成 so 文件

mkdir PokerHandEvaluator-master/cpp/build

cd PokerHandEvaluator-master/cpp/build && cmake .. && make

- Number of infostates 
  all cards:
    $$C_52^2 \cdot C_50^2 \cdot C_48^5 = 2e12$$
  all actions sequences:
    $$4209$$
  all infostates:
    $$1e16$$


