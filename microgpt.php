<?php
ini_set('memory_limit', 1024*1024*1024*300); // memory_limit = 300M
set_time_limit(0);

if(!empty($_GET['debug'])) {
    while (ob_get_level()) ob_end_flush();
	@ini_set('output_buffering', 'off');
	@ini_set('zlib.output_compression', 0);
	@ini_set('implicit_flush', 1);
	ob_implicit_flush(true);
	header('X-Accel-Buffering: no');
    header('Content-Type: text/plain; charset=UTF-8');
}

// ---------------- utils ----------------
function array_zip(array $arr1, array $arr2): array {
    $result = [];
    $max = max(count($arr1), count($arr2));
    for ($i = 0; $i < $max; $i++) {
        $result[] = [ $arr1[$i] ?? null, $arr2[$i] ?? null ];
    }
    return $result;
}

function array_zip_sum(array $arr1, array $arr2): array {
    $result = [];
    $max = max(count($arr1), count($arr2));
    for ($i = 0; $i < $max; $i++) {
        /** @var Value $a */
        /** @var Value $b */
        $a = $arr1[$i];
        $b = $arr2[$i];
        $result[] = $a->add($b);
    }
    return $result;
}

function array_range0(int $len): array {
    $res = [];
    for ($i = 0; $i < $len; $i++) $res[] = $i;
    return $res;
}

function array_shuffle_inplace(array &$arr): void {
    for ($i = count($arr) - 1; $i > 0; $i--) {
        $j = (int) floor((mt_rand() / mt_getrandmax()) * ($i + 1));
        $tmp = $arr[$j];
        $arr[$j] = $arr[$i];
        $arr[$i] = $tmp;
    }
}

function rand_float(): float { return mt_rand() / mt_getrandmax(); }

function random_gauss(float $mean, float $stdDev): float {
    $u = 0.0; $v = 0.0;
    while ($u == 0.0) $u = rand_float();
    while ($v == 0.0) $v = rand_float();
    return $mean + $stdDev * sqrt(-2.0 * log($u)) * cos(2.0 * M_PI * $v);
}

function random_choices(array $population, ?array $weights, int $k = 1): array {
    if (!$population) return [];
    $n = count($population);
    if ($k <= 0) $k = 1;

    if ($weights === null) {
        $res = [];
        for ($i = 0; $i < $k; $i++) $res[] = $population[(int) floor(rand_float() * $n)];
        return $res;
    }

    if (count($weights) !== $n) throw new Exception("weights length mismatch");

    $cum = array_fill(0, $n, 0.0);
    $total = 0.0;
    for ($i = 0; $i < $n; $i++) {
        $w = $weights[$i];
        if ($w < 0) $w = 0;
        $total += $w;
        $cum[$i] = $total;
    }
    if ($total <= 0) {
        $res = [];
        for ($j = 0; $j < $k; $j++) $res[] = $population[(int) floor(rand_float() * $n)];
        return $res;
    }

    $res = [];
    for ($j = 0; $j < $k; $j++) {
        $r = rand_float() * $total;
        $lo = 0; $hi = $n - 1; $idx = $n - 1;
        while ($lo <= $hi) {
            $mid = ($lo + $hi) >> 1;
            if ($r < $cum[$mid]) { $idx = $mid; $hi = $mid - 1; }
            else { $lo = $mid + 1; }
        }
        $res[] = $population[$idx];
    }
    return $res;
}

function matrix_fill(int $rows, int $cols, $defaultValue): array {
    $m = [];
    for ($i = 0; $i < $rows; $i++) {
        $row = [];
        for ($j = 0; $j < $cols; $j++) $row[] = $defaultValue;
        $m[] = $row;
    }
    return $m;
}

function println($msg): void {
    if (is_array($msg)) $msg = json_encode($msg, JSON_UNESCAPED_UNICODE);
    echo $msg, PHP_EOL;
}

// ---------------- Value (autograd scalar) ----------------
class Value {
    public float $data;
    public float $grad = 0.0;
    /** @var Value[] */
    public array $_children;
    /** @var float[] */
    public array $_local_grads;

    public function __construct(float $data, array $children = [], array $local_grads = []) {
        $this->data = $data;
        $this->_children = $children;
        $this->_local_grads = $local_grads;
    }

    private static function asValue($x): Value {
        return ($x instanceof Value) ? $x : new Value((float)$x);
    }

    public function add($other): Value {
        $o = self::asValue($other);
        return new Value($this->data + $o->data, [$this, $o], [1.0, 1.0]);
    }

    public function mul($other): Value {
        $o = self::asValue($other);
        return new Value($this->data * $o->data, [$this, $o], [$o->data, $this->data]);
    }

    public function pow(float $p): Value {
        return new Value($this->data ** $p, [$this], [$p * ($this->data ** ($p - 1.0))]);
    }

    public function log(): Value {
        return new Value(log($this->data), [$this], [1.0 / $this->data]);
    }

    public function exp(): Value {
        $e = exp($this->data);
        return new Value($e, [$this], [$e]);
    }

    public function relu(): Value {
        return new Value(max(0.0, $this->data), [$this], [$this->data > 0.0 ? 1.0 : 0.0]);
    }

    public function sub($other): Value {
        $o = self::asValue($other);
        return new Value($this->data - $o->data, [$this, $o], [1.0, -1.0]);
    }

    public function div($other): Value { return $this->truediv($other); }

    public function truediv($other): Value {
        if (!($other instanceof Value)) return $this->mul(($other) ** -1);
        return $this->mul($other->pow(-1.0));
    }

    public function backward(): void {
        $topo = [];
        $visited = new SplObjectStorage();

        $build = function(Value $v) use (&$build, &$topo, $visited) {
            if (!$visited->contains($v)) {
                $visited->attach($v);
                foreach ($v->_children as $ch) $build($ch);
                $topo[] = $v;
            }
        };

        $build($this);
        $this->grad = 1.0;

        for ($i = count($topo) - 1; $i >= 0; $i--) {
            $v = $topo[$i];
            $n = count($v->_children);
            for ($j = 0; $j < $n; $j++) {
                $child = $v->_children[$j];
                $local = $v->_local_grads[$j];
                $child->grad += $local * $v->grad;
            }
        }
    }
}

// ---------------- model helpers ----------------
function make_param_matrix(int $nout, int $nin, float $std = 0.08): array {
    $res = [];
    for ($i = 0; $i < $nout; $i++) {
        $row = [];
        for ($j = 0; $j < $nin; $j++) $row[] = new Value(random_gauss(0.0, $std));
        $res[] = $row;
    }
    return $res;
}

function linear(array $x, array $w): array {
    $out = [];
    foreach ($w as $wo) {
        $acc = new Value(0.0);
        foreach ($wo as $i => $wi) $acc = $acc->add($wi->mul($x[$i]));
        $out[] = $acc;
    }
    return $out;
}

function softmax(array $logits): array {
    $maxVal = -INF;
    foreach ($logits as $v) $maxVal = max($maxVal, $v->data);
    $exps = [];
    foreach ($logits as $v) $exps[] = $v->sub($maxVal)->exp();
    $total = new Value(0.0);
    foreach ($exps as $e) $total = $total->add($e);
    $res = [];
    foreach ($exps as $e) $res[] = $e->div($total);
    return $res;
}

function rmsnorm(array $x): array {
    $sum = new Value(0.0);
    foreach ($x as $xi) $sum = $sum->add($xi->mul($xi));
    $ms = $sum->mul(1.0 / count($x));
    $scale = $ms->add(1e-5)->pow(-0.5);
    $res = [];
    foreach ($x as $xi) $res[] = $xi->mul($scale);
    return $res;
}

// ---------------- "main" ----------------
// Provide $input_txt somehow (file, stdin, etc.)
$input_txt = $input_txt ?? file_get_contents('input.txt'); // set this externally

$docs = [];
foreach (preg_split("/\r?\n/", $input_txt) as $str) {
    if ($str === "" || preg_match("/^ +$/", $str)) continue;
    $docs[] = preg_replace("/ +$/", "", preg_replace("/^ +/", "", $str));
}
array_shuffle_inplace($docs);
println("num docs: " . count($docs));

$uchars = [];
foreach ($docs as $str) {
    $chars = preg_split('//u', $str, -1, PREG_SPLIT_NO_EMPTY);
    for ($i = count($chars) - 1; $i >= 0; $i--) {
        if (!in_array($chars[$i], $uchars, true)) $uchars[] = $chars[$i];
    }
}
sort($uchars);
println("uchars " . json_encode($uchars, JSON_UNESCAPED_UNICODE));

$BOS = count($uchars);
$vocab_size = count($uchars) + 1;
println("vocab size: " . $vocab_size);

$n_embd = 16;
$n_head = 4;
$n_layer = 1;
$block_size = 16;
$head_dim = intdiv($n_embd, $n_head);

$state_dict = [
    'wte'     => make_param_matrix($vocab_size, $n_embd),
    'wpe'     => make_param_matrix($block_size, $n_embd),
    'lm_head' => make_param_matrix($vocab_size, $n_embd),
];

for ($i = 0; $i < $n_layer; $i++) {
    $state_dict["layer{$i}.attn_wq"] = make_param_matrix($n_embd, $n_embd);
    $state_dict["layer{$i}.attn_wk"] = make_param_matrix($n_embd, $n_embd);
    $state_dict["layer{$i}.attn_wv"] = make_param_matrix($n_embd, $n_embd);
    $state_dict["layer{$i}.attn_wo"] = make_param_matrix($n_embd, $n_embd);
    $state_dict["layer{$i}.mlp_fc1"] = make_param_matrix(4 * $n_embd, $n_embd);
    $state_dict["layer{$i}.mlp_fc2"] = make_param_matrix($n_embd, 4 * $n_embd);
}

$params = [];
foreach ($state_dict as $mat) {
    foreach ($mat as $row) foreach ($row as $p) $params[] = $p;
}
println("num params: " . count($params));

$gpt = function(int $token_id, int $pos_id, array &$keys, array &$values) use (
    &$state_dict, $n_layer, $n_head, $head_dim
): array {
    $tok_emb = $state_dict['wte'][$token_id];
    $pos_emb = $state_dict['wpe'][$pos_id];
    $x = array_zip_sum($tok_emb, $pos_emb);
    $x = rmsnorm($x);

    for ($li = 0; $li < $n_layer; $li++) {
        // attn
        $x_residual = $x;
        $x = rmsnorm($x);
        $q = linear($x, $state_dict["layer{$li}.attn_wq"]);
        $k = linear($x, $state_dict["layer{$li}.attn_wk"]);
        $v = linear($x, $state_dict["layer{$li}.attn_wv"]);
        $keys[$li][] = $k;
        $values[$li][] = $v;

        $x_attn = [];
        for ($h = 0; $h < $n_head; $h++) {
            $hs = $h * $head_dim;
            $q_h = array_slice($q, $hs, $head_dim);

            $k_h = [];
            foreach ($keys[$li] as $ki) $k_h[] = array_slice($ki, $hs, $head_dim);
            $v_h = [];
            foreach ($values[$li] as $vi) $v_h[] = array_slice($vi, $hs, $head_dim);

            $attn_logits = [];
            for ($t = 0; $t < count($k_h); $t++) {
                $sum = new Value(0.0);
                for ($j = 0; $j < $head_dim; $j++) $sum = $sum->add($q_h[$j]->mul($k_h[$t][$j]));
                $attn_logits[] = $sum->truediv(($head_dim ** 0.5));
            }
            $attn_weights = softmax($attn_logits);

            $head_out = [];
            for ($j = 0; $j < $head_dim; $j++) {
                $sum = new Value(0.0);
                for ($t = 0; $t < count($v_h); $t++) $sum = $sum->add($attn_weights[$t]->mul($v_h[$t][$j]));
                $head_out[] = $sum;
            }
            $x_attn = array_merge($x_attn, $head_out);
        }

        $x = linear($x_attn, $state_dict["layer{$li}.attn_wo"]);
        $x = array_zip_sum($x, $x_residual);

        // mlp
        $x_residual = $x;
        $x = rmsnorm($x);
        $x = linear($x, $state_dict["layer{$li}.mlp_fc1"]);
        foreach ($x as $i => $xi) $x[$i] = $xi->relu();
        $x = linear($x, $state_dict["layer{$li}.mlp_fc2"]);
        $x = array_zip_sum($x, $x_residual);
    }

    return linear($x, $state_dict['lm_head']);
};

// Adam
$learning_rate = 0.01; $beta1 = 0.85; $beta2 = 0.99; $eps_adam = 1e-8;
$m = array_fill(0, count($params), 0.0);
$v = array_fill(0, count($params), 0.0);

$num_steps = empty($_GET['steps']) ? 1000 : intval($_GET['steps']);
for ($step = 0; $step < $num_steps; $step++) {
    $doc = $docs[$step % count($docs)];
    $chars = preg_split('//u', $doc, -1, PREG_SPLIT_NO_EMPTY);

    $tokens = [$BOS];
    foreach ($chars as $ch) $tokens[] = array_search($ch, $uchars, true);
    $tokens[] = $BOS;

    $n = min($block_size, count($tokens) - 1);

    $keys = array_fill(0, $n_layer, []);
    $values = array_fill(0, $n_layer, []);

    $losses = [];
    for ($pos_id = 0; $pos_id < $n; $pos_id++) {
        $token_id = $tokens[$pos_id];
        $target_id = $tokens[$pos_id + 1];
        $logits = $gpt($token_id, $pos_id, $keys, $values);
        $probs = softmax($logits);
        $losses[] = $probs[$target_id]->log()->mul(-1.0);
    }

    $loss_sum = new Value(0.0);
    foreach ($losses as $lt) $loss_sum = $lt->add($loss_sum);
    $loss = (new Value(1.0))->truediv($n)->mul($loss_sum);

    $loss->backward();

    $lr_t = $learning_rate * (1.0 - $step / $num_steps);
    for ($i = 0; $i < count($params); $i++) {
        $p = $params[$i];
        $m[$i] = $beta1 * $m[$i] + (1.0 - $beta1) * $p->grad;
        $v[$i] = $beta2 * $v[$i] + (1.0 - $beta2) * ($p->grad ** 2);
        $m_hat = $m[$i] / (1.0 - ($beta1 ** ($step + 1)));
        $v_hat = $v[$i] / (1.0 - ($beta2 ** ($step + 1)));
        $p->data -= $lr_t * $m_hat / (sqrt($v_hat) + $eps_adam);
        $p->grad = 0.0;
    }

    println("step " . ($step + 1) . " / {$num_steps} | loss " . $loss->data);
	gc_collect_cycles();
}

// inference
$temperature = 0.5;
println("--- inference (new, hallucinated names) ---");
for ($sample_idx = 0; $sample_idx < 20; $sample_idx++) {
    $keys = array_fill(0, $n_layer, []);
    $values = array_fill(0, $n_layer, []);
    $token_id = $BOS;
    $sample = [];

    for ($pos_id = 0; $pos_id < $block_size; $pos_id++) {
        $logits = $gpt($token_id, $pos_id, $keys, $values);
        $scaled = [];
        foreach ($logits as $l) $scaled[] = $l->truediv($temperature);
        $probs = softmax($scaled);

        $weights = [];
        foreach ($probs as $p) $weights[] = $p->data;

        $token_id = random_choices(array_range0($vocab_size), $weights, 1)[0];
        if ($token_id === $BOS) break;
        $sample[] = $uchars[$token_id];
    }
    println("sample " . ($sample_idx + 1) . ": " . implode("", $sample));
}