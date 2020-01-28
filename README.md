# Introduction

After graduating from college in 2019, I conceived an idea using a project to practise what I've leart in these years. The area I choose is financial markets. In fact, I'm not familiar with this field, and I don't have any special ideas. I chose it entirely because I saw an advertisement for an online trading platform that said that more than 80% of traders on their platform would suffer losses. I don't feel shocked or frightened by such figures, but I thought of a question: if I didn't consider the transaction cost, I decided to buy or sell by tossing a coin, and ended the transaction after a fixed period of time. Repeat the transaction N several times in this way. If N is large enough, will my profit converge to 0? To me, this is already a mathematical model that can be used for trading. It is also an interesting topic to verify the effectiveness of this mathematical model in financial markets. The question is how to implement it. To verify this model, we must first implement a system to let the system automatically trade, because I don't want to repeat the experiment N times by myself, and this N may be very large.

At the time of writing this introduction, I have actually implemented this automatic trading system, which includes a model for automatically collecting financial data, a model for analyzing the data, and a model for making decisions about buying and selling behavior. You can find it in folder forex. In this project  many functions are implemented, but it is just a small project which I came across by accident. I did not do the necessary framework design for this project, which makes it presenting a flat structure. In addition, I did not consider the possibility of distributed parallel computing, which makes the system not very scalable. Finally, the three different tasks of this system come together, which increases the complexity of the project and makes engineering management difficult. Therefore, after thinking twice, I decided to split the system into three parts. The first separated project is a data transmission system, which realises the data collection, which I am currently doing. Detailed information about this project can be found in folder DTS. Besides, I will set up new projects for the other two modules when I have time.

In short, although this project did not implement my idea perfectly, it verified the feasibility of my idea. I will continue to work in the future.

