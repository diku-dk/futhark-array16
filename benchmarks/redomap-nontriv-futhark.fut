default(f32)

fun f32 inf() = 1.0/0.0

fun f32 today() = 1.683432e7

fun f32 min(f32 a, f32 b) = if(a < b) then a else b
fun f32 max(f32 a, f32 b) = if(a < b) then b else a

fun f32 minInMonth() = 43200.0

fun f32 add_months(f32 date1, f32 num_months) =
    num_months*minInMonth() + date1

fun f32 date_365(f32 t1, f32 t2) = (t2 - t1) / (minInMonth()*12.0)

fun f32 r       () = 0.03

fun f32 zc(f32 t) = exp32(-r() * date_365(t, today()))

fun {f32, f32, f32, [f32], [f32]}
main(int n_sched) =
  let sw_mat  = 30.00f32 in
  let sw_ty   = 512.0f32 in
  let maturity  = add_months(today(), 12.0*sw_mat) in
  let sw_freq   = 12.0 * sw_ty / f32(n_sched)      in
  let beg_dates = map( add_months(maturity), map(*sw_freq, map(f32, iota(n_sched))) )
  in
  let end_dates = map( fn f32 (f32 beg_date) => add_months(beg_date, sw_freq), beg_dates)
  in
  let lvls = map ( fn f32 (f32 a1, f32 a2) => zc(a2) * date_365(a2, a1)
                 , zip(beg_dates,end_dates) )
  in
  let lvl = reduceComm(+, 0.0, lvls) in
  let t0  = reduceComm(min, inf(), beg_dates) in
  let tn  = reduceComm(max, 0.0,   end_dates)
  in
  let vt_ends    = map( zc, beg_dates ) in
  let fact_aicis = map( zc, end_dates ) in
  {lvl, t0, tn, vt_ends, fact_aicis}



