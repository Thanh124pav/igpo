# IGPO


## Introduction 
GRPO thì đánh trọng số theo trajectory-level -> thô (coarse-grained) -> có thể bỏ lỡ nhiều reasoning tốt trong 1 response sai \
PPO thì đánh trọng số theo token-level -> đòi hỏi reward model tốt, nhưng thường ko đảm bảo được và trở thành inaccurate estimation \
Chúng tôi muốn hướng tới một giải thuật segment-level, có thể credit assignment tốt hơn GRPO, nhưng sử dụng estimation (MC sampling) hiệu quả hơn PPO 


## Related works
### GRPO 
GRPO (Group Relative Policy Optimization) là một phương pháp RL finetuning cho LLMs ở mức **trajectory / response level**. Thay vì học một critic để ước lượng value như PPO, GRPO dùng nhiều responses được sample từ cùng một prompt để tạo baseline tương đối trong group.

Với một prompt $q$, policy cũ sinh ra một nhóm gồm $G$ responses:

$$
y_i \sim \pi_{\theta_{\mathrm{old}}}(\cdot \mid q),
\quad i = 1, ..., G
$$

Mỗi response được chấm điểm bằng reward function hoặc verifier:

$$
r_i = R(q, y_i)
$$

Nếu response tốt hơn trung bình group, $\hat{A}_i > 0$; nếu kém hơn trung bình group, $\hat{A}_i < 0$.


Ưu điểm của GRPO:
- không cần critic model
- đơn giản hơn PPO trong setup LLM
- group baseline giúp giảm variance so với dùng raw reward
- phù hợp với các task có final reward rõ ràng như math correctness

Hạn chế chính của GRPO là credit assignment còn thô. Mỗi response chỉ nhận một scalar advantage:

$$
\hat{A}(y_i) = \hat{A}_i
$$

Do đó các token trong cùng response thường nhận cùng một tín hiệu:

$$
\sum_{t=1}^{|y_i|}
\hat{A}_i
\log \pi_\theta(y_{i,t} \mid q, y_{i,<t})
$$

Điều này gây khó cho reasoning:
- một response sai vẫn có thể chứa nhiều reasoning steps đúng
- một response đúng vẫn có thể chứa đoạn dư thừa hoặc reasoning kém
- GRPO không biết segment nào trong response tạo ra improvement

### SPO
SPO (Search / Self-Play Optimization) là một hướng mở rộng từ policy optimization thông thường sang **tree-level reasoning optimization**.

Thay vì chỉ sample một completion hoàn chỉnh cho mỗi prompt, SPO xây dựng một cây reasoning:
- mỗi node biểu diễn một prefix / partial reasoning trace
- mỗi cạnh biểu diễn một reasoning segment được sinh tiếp từ prefix hiện tại
- mỗi path từ root tới leaf là một lời giải hoàn chỉnh

Với một prompt $q$, policy $\pi_\theta$ sinh ra nhiều reasoning branches:
$$
x_0 = q,\quad
x_{d+1} = x_d \oplus a_d,\quad
a_d \sim \pi_\theta(\cdot \mid x_d)
$$

Trong đó:
- $x_d$ là reasoning prefix tại depth $d$
- $a_d$ là segment được sinh tại depth $d$
- $x_{d+1}$ là prefix mới sau khi nối thêm segment $a_d$

Sau khi tree được construct, các leaf nodes được chấm reward bằng reward function:
$$
R(\tau) = r(x_D)
$$

Trong đó $\tau = (x_0, a_0, x_1, a_1, ..., x_D)$ là một reasoning trajectory hoàn chỉnh.

Từ reward ở leaf, SPO ước lượng value cho các internal nodes bằng cách backup reward từ các descendants:
$$
V(x_d)
\approx
\mathbb{E}_{\tau \sim \pi_\theta(\cdot \mid x_d)}
\left[
R(\tau)
\right]
$$

Trong thực tế, expectation này được xấp xỉ bằng trung bình reward của các rollouts đi qua node đó:
$$
\hat{V}(x_d)
=
\frac{1}{|\mathcal{L}(x_d)|}
\sum_{\ell \in \mathcal{L}(x_d)}
R(\ell)
$$

Với $\mathcal{L}(x_d)$ là tập leaf descendants của node $x_d$.

Sau đó, mỗi edge / segment có thể được gán advantage:
$$
A(x_d, a_d)
=
\hat{V}(x_{d+1})
-
\hat{V}(x_d)
$$

Advantage này cho biết segment $a_d$ làm tăng hay giảm expected final reward so với prefix cha.

Nhờ đó, SPO có thể tạo training signal dày hơn so với sequence-level RL:
- một trajectory hoàn chỉnh không chỉ cho một reward ở cuối
- mỗi intermediate segment trong tree đều có thể nhận một advantage estimate
- các reasoning steps tốt / xấu được phân biệt cục bộ hơn

Tuy nhiên, SPO có chi phí rất lớn vì phải expand nhiều node trong tree.

Nếu branching factor là $W$ và max depth là $D$, số node non-root trong cây đầy đủ là:
$$
\sum_{d=1}^{D} W^d
=
W + W^2 + ... + W^D
$$

Do đó chi phí inference / reward evaluation tăng theo cấp số nhân theo depth.

## Motivation
Chúng tôi nhận thấy rằng trong các giải thuật hướng tới việc mid-grained level, độ dài của 1 segment (node) ảnh hưởng tới chất lượng, và việc chọn độ dài này ko naive (M tokens như SPO-tree) mà dựa vào evidence nào đó (entropy, ....) sẽ giúp tăng performance lên. Việc chọn M như vậy thay đổi với từng bộ dataset (hay thâm chí là với từng question), do đó nó cũng liên quan tới max_depth ( = max_model_len/ M) trong cây, chúng tôi đề xuất 1 giải thuật phát triển từ SPO, tạo overhead ko đáng kể SPO khi max_depth nhỏ nhưng sẽ efficiency hơn nhiều khi depth tăng. Tổng kết lại contribution:
- Cho thấy cách chia segment ảnh hưởng tới performance
- Đề xuất giải thuật tạo overhead ko đáng kể với baseline (SPO-tree) khi depth nhỏ, nhưng effiency hơn khi depth tăng, performance giữ nguyên



## Method 
Thực hiện cắt tỉa theo hai hướng: 
- cắt tỉa các node có reasoning traces giống nhau (value sharing)
- cắt tỉa các node mà Value tại đó xấp xỉ với value tại cha của nó 

### Xấp xỉ TV divergence
$$
D_{\mathrm{TV}}
\left(
p(\cdot\mid x),
p(\cdot\mid y)
\right) 
= \sum_{z} | Pr(z \mid x ) - Pr(z \mid y) | \\ 

= \sum_{z \in p( . \mid x)}  | Pr(z \mid x ) - Pr(z \mid y) | 
+ \sum_{z \in p( . \mid y)}  | Pr(z \mid x ) - Pr(z \mid y) | \\
+ \sum_{z \text{ otherwise}}  | Pr(z \mid x ) - Pr(z \mid y) | \\

\approx \sum_{z \in p( . \mid x)}  | Pr(z \mid x ) - Pr(z \mid y) | 
+ \sum_{z \in p( . \mid y)}  | Pr(z \mid x ) - Pr(z \mid y) | + \epsilon \\

\approx \sum_{z \in TopK p( . \mid x)}  | Pr(z \mid x ) - Pr(z \mid y) | 
+ \sum_{z \in TopK p( . \mid y)}  | Pr(z \mid x ) - Pr(z \mid y) | + \epsilon \\
$$
 

Do đó, chúng tôi đề xuất việc xấp xỉ TV divergence giữa  $p( . \mid x)$ và $p( . \mid x)$ bằng *TopK* roll out từ  $p( . \mid x)$ và $p( . \mid x)$

#### Simulation Lemma: 
$$
\left|
V_x^\pi - V_y^\pi
\right|
\le
\frac{\gamma R_{\max}}{(1-\gamma)^2}
D_{\mathrm{TV}}
\left(
p(\cdot\mid x),
p(\cdot\mid y)
\right)
$$

Do đó, có thể đánh giá cận trên cho $\left|
V_x^\pi - V_y^\pi
\right|$ bởi các rollout từ  $p( . \mid x)$ và $p( . \mid x)$

### Prunning

#### Value sharing 
Nếu: 
$$
\left|
V_x^\pi - V_y^\pi
\right|
\le
\frac{\gamma R_{\max}}{(1-\gamma)^2}
D_{\mathrm{TV}}
\left(
p(\cdot\mid x),
p(\cdot\mid y)
\right) \\
\approx \frac{\gamma R_{\max}}{(1-\gamma)^2} \epsilon \leq \eta
$$
Thì thực hiện cắt tỉa 

#### Child prunning: 
Trong RL: 
$$ 
V^\pi(s)=\sum_i \pi(a_i\mid s)Q^\pi(s,a_i)
$$ 

Gọi $s_i$ là state thu được từ $s$ sau khi thực hiện action $a_i$ 
$$

\Rightarrow | V^{\pi}(s_i) - V^\pi(s) | = | \sum_{j \neq i} \pi(a_j \mid s) (V^{\pi}(s_i) - V^{\pi}(s_j) )  | \\ 
\leq \sum_{j \neq i}  \pi(a_j \mid s) | V^{\pi}(s_i) - V^{\pi}(s_j) | $$ 

Tiếp tục xấp xỉ  $| V^{\pi}(a_i) - V^{\pi}(a_j) |$ , nếu nhỏ hơn 1 ngưỡng $\epsilon$, ta lấy luôn $V^{\pi}(s_i) \in [ V^\pi(s) - \epsilon, V^\pi(s) + \epsilon]$ , từ đó, ko phát triển tiếp node $s_i$


### Hướng phát triển
max depth hiện đang bằng nhau với mọi prompt, nếu có thể phân bổ max depth tùy theo độ khó của câu hỏi thì có thể hướng tới 1 giải thuật budget allocation: phân bổ những cây độ sâu lớn cho các câu hỏi khó, và các cây đơn giản cho câu hỏi dễ 
