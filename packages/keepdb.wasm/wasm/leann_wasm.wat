(module
  (memory (export "memory") 1)
  (global $heap (mut i32) (i32.const 1024))

  (func (export "malloc") (param $size i32) (result i32)
    (local $ptr i32)
    (local.set $ptr (global.get $heap))
    (global.set $heap
      (i32.add
        (global.get $heap)
        (i32.and
          (i32.add (local.get $size) (i32.const 7))
          (i32.const -8))))
    (local.get $ptr))

  (func (export "free") (param $ptr i32)
    nop)

  (func (export "leann_wasm_version") (result i32)
    (i32.const 1))

  (func $load_f32 (param $ptr i32) (param $index i32) (result f32)
    (f32.load
      (i32.add
        (local.get $ptr)
        (i32.mul (local.get $index) (i32.const 4)))))

  (func $score_dot (export "leann_wasm_score_dot")
    (param $a i32)
    (param $b i32)
    (param $dimension i32)
    (result f32)
    (local $i i32)
    (local $total f32)

    (local.set $i (i32.const 0))
    (local.set $total (f32.const 0))

    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $dimension)))
        (local.set $total
          (f32.add
            (local.get $total)
            (f32.mul
              (call $load_f32 (local.get $a) (local.get $i))
              (call $load_f32 (local.get $b) (local.get $i)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))

    (local.get $total))

  (func $sqrt_approx (param $value f32) (result f32)
    (local $i i32)
    (local $x f32)

    (if (f32.le (local.get $value) (f32.const 0))
      (then (return (f32.const 0))))

    (if (f32.gt (local.get $value) (f32.const 1))
      (then (local.set $x (local.get $value)))
      (else (local.set $x (f32.const 1))))

    (local.set $i (i32.const 0))
    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (i32.const 12)))
        (local.set $x
          (f32.mul
            (f32.const 0.5)
            (f32.add
              (local.get $x)
              (f32.div (local.get $value) (local.get $x)))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))

    (local.get $x))

  (func $score_l2
    (param $a i32)
    (param $b i32)
    (param $dimension i32)
    (result f32)
    (local $i i32)
    (local $diff f32)
    (local $total f32)

    (local.set $i (i32.const 0))
    (local.set $total (f32.const 0))

    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $dimension)))
        (local.set $diff
          (f32.sub
            (call $load_f32 (local.get $a) (local.get $i))
            (call $load_f32 (local.get $b) (local.get $i))))
        (local.set $total
          (f32.add
            (local.get $total)
            (f32.mul (local.get $diff) (local.get $diff))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))

    (f32.neg (call $sqrt_approx (local.get $total))))

  (func $score_cosine
    (param $a i32)
    (param $b i32)
    (param $dimension i32)
    (result f32)
    (local $i i32)
    (local $av f32)
    (local $bv f32)
    (local $dot f32)
    (local $norm_a f32)
    (local $norm_b f32)
    (local $denom f32)

    (local.set $i (i32.const 0))
    (local.set $dot (f32.const 0))
    (local.set $norm_a (f32.const 0))
    (local.set $norm_b (f32.const 0))

    (block $done
      (loop $loop
        (br_if $done (i32.ge_s (local.get $i) (local.get $dimension)))
        (local.set $av (call $load_f32 (local.get $a) (local.get $i)))
        (local.set $bv (call $load_f32 (local.get $b) (local.get $i)))
        (local.set $dot (f32.add (local.get $dot) (f32.mul (local.get $av) (local.get $bv))))
        (local.set $norm_a (f32.add (local.get $norm_a) (f32.mul (local.get $av) (local.get $av))))
        (local.set $norm_b (f32.add (local.get $norm_b) (f32.mul (local.get $bv) (local.get $bv))))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $loop)))

    (if (f32.le (local.get $norm_a) (f32.const 0))
      (then (return (f32.const 0))))
    (if (f32.le (local.get $norm_b) (f32.const 0))
      (then (return (f32.const 0))))

    (local.set $denom
      (f32.mul
        (call $sqrt_approx (local.get $norm_a))
        (call $sqrt_approx (local.get $norm_b))))
    (f32.div (local.get $dot) (local.get $denom)))

  (func $score_vector
    (param $a i32)
    (param $b i32)
    (param $dimension i32)
    (param $metric i32)
    (result f32)

    (if (i32.eq (local.get $metric) (i32.const 1))
      (then (return (call $score_cosine (local.get $a) (local.get $b) (local.get $dimension)))))

    (if (i32.eq (local.get $metric) (i32.const 2))
      (then (return (call $score_l2 (local.get $a) (local.get $b) (local.get $dimension)))))

    (call $score_dot (local.get $a) (local.get $b) (local.get $dimension)))

  (func (export "leann_wasm_flat_search")
    (param $vectors i32)
    (param $vector_count i32)
    (param $dimension i32)
    (param $query i32)
    (param $top_k i32)
    (param $metric i32)
    (param $out_labels i32)
    (param $out_scores i32)
    (result i32)

    (local $i i32)
    (local $label i32)
    (local $rank i32)
    (local $insert_at i32)
    (local $candidate i32)
    (local $candidate_score f32)

    (if
      (i32.or
        (i32.or (i32.eqz (local.get $vectors)) (i32.eqz (local.get $query)))
        (i32.or (i32.eqz (local.get $out_labels)) (i32.eqz (local.get $out_scores))))
      (then (return (i32.const -1))))

    (if
      (i32.or
        (i32.or (i32.le_s (local.get $vector_count) (i32.const 0)) (i32.le_s (local.get $dimension) (i32.const 0)))
        (i32.le_s (local.get $top_k) (i32.const 0)))
      (then (return (i32.const -2))))

    (local.set $i (i32.const 0))
    (block $init_done
      (loop $init_loop
        (br_if $init_done (i32.ge_s (local.get $i) (local.get $top_k)))
        (i32.store
          (i32.add (local.get $out_labels) (i32.mul (local.get $i) (i32.const 4)))
          (i32.const -1))
        (f32.store
          (i32.add (local.get $out_scores) (i32.mul (local.get $i) (i32.const 4)))
          (f32.const -3.402823e38))
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $init_loop)))

    (local.set $label (i32.const 0))
    (block $search_done
      (loop $search_loop
        (br_if $search_done (i32.ge_s (local.get $label) (local.get $vector_count)))

        (local.set $candidate
          (i32.add
            (local.get $vectors)
            (i32.mul
              (i32.mul (local.get $label) (local.get $dimension))
              (i32.const 4))))
        (local.set $candidate_score
          (call $score_vector
            (local.get $query)
            (local.get $candidate)
            (local.get $dimension)
            (local.get $metric)))

        (local.set $insert_at (i32.const -1))
        (local.set $rank (i32.const 0))
        (block $rank_done
          (loop $rank_loop
            (br_if $rank_done (i32.ge_s (local.get $rank) (local.get $top_k)))
            (if
              (f32.gt
                (local.get $candidate_score)
                (f32.load
                  (i32.add
                    (local.get $out_scores)
                    (i32.mul (local.get $rank) (i32.const 4)))))
              (then
                (local.set $insert_at (local.get $rank))
                (br $rank_done)))
            (local.set $rank (i32.add (local.get $rank) (i32.const 1)))
            (br $rank_loop)))

        (if (i32.ge_s (local.get $insert_at) (i32.const 0))
          (then
            (local.set $rank (i32.sub (local.get $top_k) (i32.const 1)))
            (block $shift_done
              (loop $shift_loop
                (br_if $shift_done (i32.le_s (local.get $rank) (local.get $insert_at)))
                (f32.store
                  (i32.add (local.get $out_scores) (i32.mul (local.get $rank) (i32.const 4)))
                  (f32.load
                    (i32.add
                      (local.get $out_scores)
                      (i32.mul (i32.sub (local.get $rank) (i32.const 1)) (i32.const 4)))))
                (i32.store
                  (i32.add (local.get $out_labels) (i32.mul (local.get $rank) (i32.const 4)))
                  (i32.load
                    (i32.add
                      (local.get $out_labels)
                      (i32.mul (i32.sub (local.get $rank) (i32.const 1)) (i32.const 4)))))
                (local.set $rank (i32.sub (local.get $rank) (i32.const 1)))
                (br $shift_loop)))

            (f32.store
              (i32.add (local.get $out_scores) (i32.mul (local.get $insert_at) (i32.const 4)))
              (local.get $candidate_score))
            (i32.store
              (i32.add (local.get $out_labels) (i32.mul (local.get $insert_at) (i32.const 4)))
              (local.get $label))))

        (local.set $label (i32.add (local.get $label) (i32.const 1)))
        (br $search_loop)))

    (i32.const 0))
)
