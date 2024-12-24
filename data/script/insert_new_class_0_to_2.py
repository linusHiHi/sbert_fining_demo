import pandas as pd

add0 = ["我得去车站，先确认一下天气怎么样", "我打算去车站，得看看外面天气如何",
"我要去车站，先看看天气预报", "去车站之前，我得先确认一下天气", "我准备去车站，得了解一下天气情况",
"去车站前，先查查今天的天气", "我要去车站，得知道外面天气如何",
"我去车站前得先看看天气变化", "我要去车站，最好先确认一下是否会下雨",
"去车站之前，我得看看天气是否适合出门"]

add1 = ["今天外面下雨，我正好想在家里吃点热饭", "天气有点冷，正好想吃点热乎乎的火锅",
"外面刮风了，我正好想在餐馆里吃顿饭", "今天太热了，想去吃点清爽的沙拉", "外面下着小雨，正是想吃碗热汤的时候",
"天气不好，我更想在家吃个温暖的晚餐", "今天阴天，正适合吃些热菜暖胃", "外面刮风很冷，正好想去吃点热面",
"今天雨下得很大，我想去吃碗热乎乎的面条", "天气冷了，我想去吃点温暖的食物",
        "我去车站是为了买票，但其实我打算先在附近的餐馆吃点东西",
               "我得去车站买票，不过先去车站旁边的餐厅吃点饭", "去车站买票顺便在附近的小吃店尝尝",
               "我要去车站取票，不过先去车站附近的咖啡店坐坐", "我先去车站买票，顺便去旁边的面馆吃碗面",
               "我去车站是为了取票，但计划在附近的餐厅吃个午餐", "我去车站的目的是买票，但我先在周围的小吃街吃点东西",
               "我得去车站买票，然后再去站旁的餐馆吃点饭", "我去车站买票，顺便去附近的包子铺吃个早餐",
               "我打算去车站买票，不过先去车站附近的快餐店吃点东西"]
add2 = ["外面天气不好，我还是打算在线上买票", "今天下雨太大，还是直接网上购票吧",
               "天气这么糟糕，不如在网上买票更方便", "外面风很大，我决定在线上买票",
               "今天的天气不适合出门，我还是通过网上买票", "外面下雪了，还是在网上买票比较安全",
               "暴雨天气下，还是在线上买票比较省事", "今天的天气太差了，不想去车站买票，直接网上购票",
               "外面风雨交加，干脆网上购票算了", "天气不好，线上买票更加方便快捷"]

qwq = pd.concat(
    [
        pd.DataFrame({"sentence":add0, "class": [0]*len(add0)}),
        pd.DataFrame({"sentence":add1, "class": [1]*len(add1)}),
        pd.DataFrame({"sentence":add2, "class": [2]*len(add2)})
    ]
)
qwq.to_csv("../data/found_class_0_to_2.csv",index=False)