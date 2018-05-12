---
title: python实践记录
date: 2018-04-28 00:24:01
tags:
---

本文主要记录python实践过程中的一些知识点。

<!-- more -->

**os模块**

os.walk(path) 得到

输入：文件夹路径

返回：三元tupple依次是(dirpath,dirnames,filenames)

os.path.basename()

输入：文件路径

输出：当前文件夹名

os.paht.dirname

输入：文件路径

输出：根目录名

```
import os
path = r'D:\path'
print(list(os.walk(path)))
out:
[('D:\\path', ['to'], ['FILEINto.txt']),
('D:\\path\\to', ['MNIST_data'], ['fileInMNIST.rar']), 
('D:\\path\\to\\MNIST_data', [], ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'])]
```

打印文件夹结构

```
def fileCntIn(currPath):    
    #计算总路径内文件数，通过将每个文件夹内文件数相加
    return sum([len(files) for root, dirs, files in os.walk(currPath)])    
    
def dirsTree(startPath):    
    '''''''树形打印出目录结构'''    
    for root, dirs, files in os.walk(startPath):    
        #获取当前目录下文件数    
        fileCount = fileCntIn(root)    
        #获取当前目录相对输入目录的层级关系,整数类型，os.sep为跨平台分割符    
        level = root.replace(startPath, '').count(os.sep)    
        #树形结构显示关键语句    
        #根据目录的层级关系，重复显示'| '间隔符，    
        #第一层 '| '    
        #第二层 '| | '    
        #第三层 '| | | '    
        #依此类推...    
        #在每一层结束时，合并输出 '|____'    
        indent = '| ' * 1 * level + '|____'    
        print（'%s%s -r:%s' % (indent, os.path.split(root)[1], fileCount))
        for file in files:  
            indent = '| ' * 1 * (level+1) + '|____'    
            print('%s%s' % (indent, file))   
    
if __name__ == '__main__':    
	dirsTree(path)  
out:
|____path -r:6
| |____FILEINto.txt
| |____to -r:5
| | |____fileInMNIST.rar
| | |____MNIST_data -r:4
| | | |____t10k-images-idx3-ubyte.gz
| | | |____t10k-labels-idx1-ubyte.gz
| | | |____train-images-idx3-ubyte.gz
| | | |____train-labels-idx1-ubyte.gz
```

**glob模块**

glob 查找符合特定规则的文件路径名
glob.glob(path)

输入文件夹路径

返回特定文件路径

glob.iglob(path)

返回迭代器

```
import os
path = r'D:\path\to\MNIST_data\*.rar'
print(glob.glob(path))
out:
['D:\\path\\to\\MNIST_data\\t10k-images-idx3-ubyte.gz', 'D:\\path\\to\\MNIST_data\\t10k-labels-idx1-ubyte.gz', 'D:\\path\\to\\MNIST_data\\train-images-idx3-ubyte.gz', 'D:\\path\\to\\MNIST_data\\train-labels-idx1-ubyte.gz']
```

