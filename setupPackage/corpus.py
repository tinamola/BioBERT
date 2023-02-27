import xml.etree.ElementTree as ET
import os, gzip, shutil,sys,glob,re
import re
from urllib import request
import gzip
import shutil
#
# url1 = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n0005.xml.gz"
# file_name1 = re.split(pattern='/', string=url1)[-1]
# r1 = request.urlretrieve(url=url1, filename=file_name1)
# txt1 = re.split(pattern=r'\.', string=file_name1)[0] + ".txt"
# print(file_name1,txt1)
# with gzip.open(file_name1, 'rb') as f_in:
#     with open(txt1, 'wb') as f_out:
#         shutil.copyfileobj(f_in, f_out)
from datasets import DownloadManager as dl_manager
# urls=[f"https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/pubmed22n{i:04d}.xml.gz" for i in range(1, 1115)]
#https://gist.github.com/kstreepy/a9800804c21367d5a8bde692318a18f5
# def gz_extract(directory):
#     extension = ".gz"
#     os.chdir(directory)
#     for item in os.listdir(directory):  # loop through items in dir
#         if item.endswith(extension):  # check for ".gz" extension
#             try:
#                 gz_name = os.path.abspath(item)  # get full path of files
#                 file_name = (os.path.basename(gz_name)).rsplit('.', 1)[0]  # get file name for file within
#                 with gzip.open(gz_name, "rb") as f_in, open(file_name, "wb") as f_out:
#                     shutil.copyfileobj(f_in, f_out)
#                 os.remove(gz_name)  # delete zipped file
#             except:
#                 continue
#
# def xml_to_txt(indir,outdir=None):
#     os.chdir(indir)  # indir为xml文件来源的文件夹，outdir为转换的txt文件存储路径
#     annotated = [f for f in os.listdir('.') if re.match(r'[\w]*.xml$', f)]  # 返回包含目录中文件名称的列表
#     for i, file in enumerate(annotated):
#         try:
#             file_save = file.split('.')[0] + '.txt'  # split将文件名与后缀名划分开来
#             file_txt = outdir + "/" + file_save
#             f_w = open(file_txt, 'w',encoding="utf-8")
#
#             in_file = open(file, encoding='UTF-8')
#             tree = ET.parse(in_file)
#             root = tree.getroot()
#
#             for value in root.iter("AbstractText"):
#                 string=ET.tostring(value, encoding='utf8', method='text').decode("utf-8").strip()+"\n"
#                 # print(string)
#                 f_w.write(string)
#             f_w.close()
#         except:
#             print(file)
#             continue

from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(r"D:\OneDrive\Desktop\FIT4441\LAMA\pre-trained_language_models\bert\cased_L-12_H-768_A-12")
# normal method, can't strip \n
# csvfile = open(r'D:\OneDrive\Desktop\bookcorpus\bookcorpus.csv', encoding='utf-8')
# with open(r'D:\OneDrive\Desktop\bookcorpus\bookcorpus1.csv', "w+",encoding='utf-8') as csv_file1,open(r'D:\OneDrive\Desktop\bookcorpus\bookcorpus0.csv', "w+",encoding='utf-8') as csv_file2:
#     for row in csvfile:
                # row=row.strip("\n")
                # print(" ".join(tokenizer.tokenize(row)))
                # csv_file2.write(row)
                # csv_file1.write(" ".join(tokenizer.tokenize(row)))
directory=r'D:\OneDrive\Desktop\PubMed Central Full Texts\PMC000xxxxxx'
os.chdir(directory)
with open(r'D:\OneDrive\Desktop\PubmedAbstract\bioneat.txt', "w+") as csv_file1:
    for item in os.listdir(directory):  # loop through items in dir
        print(item)
        f_w = open(item, 'r', encoding="utf-8")
        for row in f_w:
            row=row.strip("\n")
            print(row)

# import csv
# csv_file = r'D:\OneDrive\Desktop\bookcorpus\bookcorpus.csv'
# txt_file = r'D:\OneDrive\Desktop\bookcorpus\bookcorpus.txt'
# with open(txt_file, "w",encoding='utf-8') as my_output_file:
#     with open(csv_file, "r",encoding='utf-8') as my_input_file:
#         for row in my_input_file:
#             if row != "\n":
#                 row = row.strip("\n")
#                 my_output_file.write(row)
#     my_output_file.close()
# for line in open(r'D:\OneDrive\Desktop\bookcorpus\bookcorpus.csv', encoding='utf-8'):
#     print(line.strip("\n"))
#     newline=tokenizer.tokenize(line)
#     print(newline)
#     a+=1
#     if a==10:
#         break

# this will ignore all \n, and i is the whole thing, but have bugs in it
# chunksize = 1
# df=pd.read_csv(r'D:\OneDrive\Desktop\bookcorpus\test.csv',header=None,index_col=False)
# df=df.ix[:,0]
# for row in df.iterrows():
#     print(row)
    # print(" ".join(tokenizer.tokenize(row)))

# print(len(df))
    # for i in chunk:
    #     print(tokenizer.tokenize(i))
    #     break
    # break
# with open(r'D:\OneDrive\Desktop\bookcorpus\bookcorpus.csv',"r",encoding="utf-8") as csvfile:
#     csv_reader = csv.reader(csvfile)
#     with open(r'D:\OneDrive\Desktop\bookcorpus\bookcorpus1.csv', 'w') as new_file:
#         for line in csv_reader:
#             print(line)
#             line.strip("\n")
#             new_file.write(line)

#this is an recurrsion still don't solve \n
# bigfile = open(r'D:\OneDrive\Desktop\bookcorpus\bookcorpus.csv', 'r', encoding='utf-8')
# tmp_lines = bigfile.readlines(100)
# a=0
# while tmp_lines:
#     for line in tmp_lines:
#         newline = tokenizer.tokenize(line)
#         print(newline)
#     tmp_lines = bigfile.readlines(100)
#     a+=1
#     if a==3:
#         break

# def getstuff(filename):
#     with open(r'D:\OneDrive\Desktop\bookcorpus\bookcorpus.csv',"r",encoding="utf-8") as csvfile:
#         datareader = csv.reader(csvfile)
#         yield next(datareader)  # yield the header row
# a=0
# for row in getstuff(r'D:\OneDrive\Desktop\bookcorpus\bookcorpus.csv'):
#     a+=1
#     print(row)
#     # newline = tokenizer.tokenize(row)
#     # print(newline)
#     if a==1:
#         break


# gz_extract(r"D:\OneDrive\Desktop\PubmedAbstract\base")
# xml_to_txt(r"D:\OneDrive\Desktop\PubmedAbstract\abstractTxT",'D:/OneDrive/Desktop/PubmedAbstract/abstractTxT')
# xml_to_txt(r"D:\OneDrive\Desktop\PubmedAbstract\baseline",'D:/OneDrive/Desktop/PubmedAbstract/abstractTxT')

from pytorch_pretrained_bert import BertTokenizer
import os
tokenizer = BertTokenizer.from_pretrained(r"LAMA/pre-trained_language_models/bert/cased_L-12_H-768_A-12")
directory="BioBERTcorpus"
os.chdir(directory)
with open(r'neatALL.txt', "w+") as csv_file1,open(r'bpeALL.txt', "w+") as csv_file2:
    for item in os.listdir("../BioBERTcorpus"):
        f_w=open(item,'rb')
        for row in f_w:
            if row!="\n":
                    row=row.strip("\n")
                    row1=" ".join(tokenizer.basic_tokenizer.tokenize(row))
                    row2=" ".join(tokenizer.tokenize(row))
                    row1+='\n'
                    row2+='\n'
                    csv_file1.write(row1)
                    csv_file2.write(row2)
