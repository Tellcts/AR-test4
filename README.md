# Steps

1. **安装`UV`工具**
- Linux Or MacOS:
  ~~~bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ~~~
- Windows:
  ~~~powershell
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
  ~~~

2. **克隆仓库**
   ~~~bash
   git clone --depth=1 https://github.com/Tellcts/AR-test4.git
   ~~~

3. **运行**
   ~~~bash
   cd AR-test4 && uv sync
   # 使用 OpenCV 进行实验
   uv run opencv/mnist.py
   uv run opencv/cifar10.py

   # 使用 CNN 进行实验
   uv run cnn/mnist.py
   uv run cnn/cifar10.py
   ~~~

4. Tips
   > - 首次运行时会进行数据集的下载，因此可能比较耗费时间，耐心等待即可；
   > - 大家也可自定提供数据集，只需将数据集放在`data`目录下；
   > - eg.`AR-test4/data/mnist/raw/`,`cifaf-10-python.tar.gz`解压至`AR-test4/data/。