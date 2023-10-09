## 说明

本项目包含下面的内容：

- 源代码：
  在README.md所在的目录下找到GEMMacc/src，在该文件夹中，final.cpp为报告提到的最终优化的代码，gemmAcc.cpp包含了前面所有的优化方法，三个testxxx.cpp文件分别代表三个线性代数计算库的调用程序。

- 可执行文件：
  在README.md所在的目录下找到GEMMacc/bin，在该文件夹中，文件名xxx即对应src文件夹下的xxx.cpp的编译结果。

#### 如何编译程序

对于下面的所有编译指令，建议加上编译优化-O3。

对于gemmAcc.cpp，请使用下面的编译指令：

```shell
g++ gemmAcc.cpp -o gemmAcc -mavx512f
```

对于final.cpp，请使用下面的编译指令：

```shell
g++ final.cpp -o final -mavx512f
```

对于testEigen.cpp，请使用下面的编译指令：

```shell
g++ testEigen.cpp -o testEigen
```

对于testarma.cpp，请使用下面的编译指令：

```shell
g++ testarma.cpp -o testarma -larmadillo
```

对于testopenblas.cpp，请使用下面的编译指令：

```shell
g++ testopenblas.cpp -o testopenblas -lopenblas
```

#### 如何执行程序

对于final，它接收三个参数作为m，n，k作为两个矩阵的规模大小。

对于其他程序，它们均接收一个参数作为方阵规模大小，其中gemmAcc接收32的倍数。

#### 程序执行结果

所有程序，将会打印GEMM执行时间（毫秒），其中gemmAcc将会连续打印多种优化方案的结果。
