'''
this file contains all the constents reqired for
our code. this makes the whole thing much more easy.

you do not need to read this you will get to know about
them when you read the code eventualy.
'''


# training parameters
th = 30 
lr =0.01
epochs = 2
threshold = 30
interval = 100

# buying and selling threshold
selling_threshold = .535
buying_threshold = .475

# graph points
x_buyingprice = []
y_buyingprice = []

x_sellingprice = []
y_sellingprice = []

# data parameters
company = 'goog'
features = ['open','high','low','close']
