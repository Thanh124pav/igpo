# InGPO Method

Tài liệu này mô tả phần công thức lý thuyết cho hai cơ chế chính của InGPO:

- siblings value sharing
- child pruning from parent

Ký hiệu:

- `u` là node cha ở depth `d - 1`.
- `v` là một node con của `u` ở depth `d`.
- `S(u)` là tập siblings được sinh ra từ parent `u`.
- `W_d` là branching factor từ depth `d` sang depth `d + 1`.
- `D` là max depth của tree.
- `K(x)` là tập rollout/local continuations được sample từ node `x`.
- `m` là số rollout dùng để ước lượng local value distribution.
- `V(x)` là value estimate của node `x`.
- `P_x` là empirical value distribution tại node `x`.
- `tau_d` là threshold cho siblings value sharing ở depth `d`.
- `eta_d` là threshold cho child pruning from parent ở depth `d`.
- `A(v)` là advantage gán cho edge `u -> v`.

## Value Estimate

Với một node `x`, InGPO ước lượng value bằng reward trung bình của các rollout từ node đó:

$$
V(x)
=
\frac{1}{m}
\sum_{k = 1}^{m}
r(x, y_k)
$$

Trong đó `y_k` là continuation thứ `k` được sample từ `x`, và `r(x, y_k)` là final reward của continuation đó.

Empirical value distribution tại node `x` là:

$$
P_x
=
\{
r(x, y_1),
r(x, y_2),
...,
r(x, y_m)
\}
$$

Advantage mặc định của child `v` so với parent `u` là:

$$
A(v)
=
V(v)
-
V(u)
$$

## Siblings Value Sharing

Trong SPO đầy đủ, khi expand parent `u`, thuật toán sẽ evaluate toàn bộ siblings:

$$
S(u)
=
\{
v_1,
v_2,
...,
v_W
\}
$$

InGPO tránh evaluate lại một child nếu child đó đủ giống một sibling đã được evaluate trước đó.

Gọi `E(u)` là tập siblings dưới parent `u` đã được evaluate. Với candidate child `v_j`, sibling gần nhất được định nghĩa là:

$$
s^*(v_j)
=
\arg\min_{s \in E(u)}
D(P_{v_j}, P_s)
$$

Trong đó `D` là khoảng cách giữa hai empirical value distributions, ví dụ total variation distance.

Điều kiện để share value là:

$$
\min_{s \in E(u)}
D(P_{v_j}, P_s)
\le
\tau_d
$$

Nếu điều kiện này đúng, InGPO gán action của `v_j` là `share` và dùng lại value của sibling gần nhất:

$$
V(v_j)
:=
V(s^*(v_j))
$$

Advantage của node shared vẫn được tính so với parent thật của nó:

$$
A(v_j)
=
V(s^*(v_j))
-
V(u)
$$

Về mặt lý thuyết, `v_j` không bị xem là kém. Nó chỉ được xem là redundant với một sibling đã biết value distribution. Vì vậy node `v_j` vẫn là factual node và vẫn có thể emit edge/training example.

Phần bị skip là descendants của `v_j`:

$$
\text{Desc}(v_j)
\text{ không được expand}
$$

Số node bị share-pruned bởi một shared node ở depth `d` là kích thước estimated descendant subtree:

$$
\text{share_pruned}(v_j)
=
\sum_{\ell = d + 1}^{D}
\prod_{t = d}^{\ell - 1}
W_t
$$

Node `v_j` không nằm trong `share_pruned`, vì nó vẫn là factual node đã được tạo ra.

## Child Pruned From Parent

Sau khi có value estimate của child, InGPO so sánh child với parent của nó.

Parent-child value gap là:

$$
G(u, v)
=
|V(v) - V(u)|
$$

Điều kiện prune child từ parent là:

$$
G(u, v)
\le
\eta_d
$$

Nếu điều kiện này đúng, InGPO gán action của `v` là `prune`.

Ý nghĩa của prune ở đây không phải là child `v` tệ. Nó chỉ có nghĩa là value của `v` không khác parent `u` đủ nhiều để justify việc expand cả subtree bên dưới `v`.

Advantage lý thuyết của node pruned vẫn là:

$$
A(v)
=
V(v)
-
V(u)
$$

Với default hiện tại:

$$
\text{zero_advantage_when_pruned}
=
\text{false}
$$

Do đó advantage đưa vào training vẫn được giữ nguyên:

$$
A_{\text{train}}(v)
=
A(v)
$$

Nếu bật `zero_advantage_when_pruned`, advantage training sẽ bị set về zero:

$$
A_{\text{train}}(v)
=
0
$$

Nhưng default đúng hơn về mặt lý thuyết là giữ advantage, vì node bị prune do nó gần parent, không phải vì nó là action xấu.

Với node `v` bị prune ở depth `d`, toàn bộ subtree rooted at `v` bị skip. Khác với siblings value sharing, bản thân node `v` được tính vào prune count:

$$
\text{pruned}(v)
=
1
+
\sum_{\ell = d + 1}^{D}
\prod_{t = d}^{\ell - 1}
W_t
$$

Số `1` ở công thức trên chính là factual child node `v`.

## Prune Rate Accounting

Không nên tính denominator bằng công thức full tree:

$$
\sum_{\ell = 1}^{D}
\prod_{t = 0}^{\ell - 1}
W_t
$$

Lý do là trong thực tế một số branch có thể terminate sớm do sinh EOS. Nếu vẫn dùng full `W^D`, denominator sẽ bị phóng đại so với tree mà SPO thật sự construct được.

Thay vào đó, InGPO dùng:

$$
\text{spo_node_count}
=
\text{factual_node_count}
+
\text{virtual_pruned_spo_count}
$$

Trong đó:

$$
\text{factual_node_count}
=
\#\{
\text{constructed non-root nodes}
\}
$$

và:

$$
\text{virtual_pruned_spo_count}
=
\sum_{v \in \mathcal{P}}
(
\text{pruned}(v)
-
1
)
+
\sum_{v \in \mathcal{S}}
\text{share_pruned}(v)
$$

Ở đây `P` là tập node có action `prune`, còn `S` là tập node có action `share`.

Prune rate được tính là:

$$
\text{prune_rate}
=
\frac{
\sum_{v \in \mathcal{P}}
\text{pruned}(v)
}{
\text{spo_node_count}
}
$$

Share prune rate được tính là:

$$
\text{share_prune_rate}
=
\frac{
\sum_{v \in \mathcal{S}}
\text{share_pruned}(v)
}{
\text{spo_node_count}
}
$$

Total prune rate là:

$$
\text{total_prune_rate}
=
\frac{
\sum_{v \in \mathcal{P}}
\text{pruned}(v)
+
\sum_{v \in \mathcal{S}}
\text{share_pruned}(v)
}{
\text{spo_node_count}
}
$$

Cách tính này giữ denominator gần với tree mà SPO sẽ construct trong cùng điều kiện early termination, đồng thời vẫn credit đúng số virtual nodes mà InGPO đã skip nhờ prune và value sharing.

## Ví dụ

Giả sử `W = 4`, `D = 2`, và một node ở depth `1` bị `prune`.

Số node bị prune là:

$$
\text{pruned}(v)
=
1
+
4
=
5
$$

Nếu các branch còn lại phát triển bình thường, `spo_node_count = 20`, nên:

$$
\text{prune_rate}
=
\frac{5}{20}
=
25\%
$$

Giả sử một node ở depth `1` bị `share`.

Node shared vẫn là factual node, chỉ có `4` children ở depth `2` bị skip:

$$
\text{share_pruned}(v)
=
4
$$

Nếu `spo_node_count = 20`, thì:

$$
\text{share_prune_rate}
=
\frac{4}{20}
=
20\%
$$

Nếu một node depth `1` bị `prune` và một node depth `1` khác bị `share`, thì:

$$
\text{total_prune_rate}
=
\frac{5 + 4}{20}
=
45\%
$$
