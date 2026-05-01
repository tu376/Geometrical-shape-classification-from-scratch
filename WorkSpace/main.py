from datetime import datetime

def parse_time(t):
    return datetime.strptime(t, "%H:%M:%S")

lst = []
while True:
    String = input().split()
    if String[0] == "#":
        break
    CustomerID, ProductID, Price, ShopID, TimePoint = String
    lst.append((CustomerID, ProductID, int(Price), ShopID, TimePoint))

while True:
    String = input().split()
    if String[0] == "#":
        break
    elif String[0] == "?total_number_orders":
        print(len(lst))
    elif String[0] == "?total_revenue":
        res = sum(item[2] for item in lst)
        print(res)
    elif String[0] == "?revenue_of_shop":
        tmp = String[1]
        res = sum(item[2] for item in lst if item[3] == tmp)
        print(res)
    elif String[0] == "?total_consume_of_customer_shop":
        CustomerID, ShopID = String[1], String[2]
        res = sum(item[2] for item in lst if item[0] == CustomerID and item[3] == ShopID)
        print(res)
    elif String[0] == "?total_revenue_in_period":
        from_time = parse_time(String[1])
        to_time = parse_time(String[2])
        res = sum(item[2] for item in lst if from_time <= parse_time(item[4]) <= to_time)
        print(res)