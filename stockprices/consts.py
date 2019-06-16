'''
this file contains all the constents reqired for
our code. this makes the whole thing much more easy.

you do not need to read this you will get to know about
them when you read the code eventualy.
'''


# training parameters
th = 10
lr =0.01
epochs = 2
threshold = 10
interval = 500

# buying and selling threshold
selling_threshold = .525
buying_threshold = .475

# graph points
x_buyingprice = []
y_buyingprice = []

x_sellingprice = []
y_sellingprice = []

# data parameters
company = 'amzn'
features = ['open','high','low','close']
