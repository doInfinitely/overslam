@echo off
:: Mei Cartographer launcher -- always uses Python 3.12.
:: Usage: cartographer.bat --map kingsrow --negatives D:\negatives
py -3.12 mei_cartographer.py %*
