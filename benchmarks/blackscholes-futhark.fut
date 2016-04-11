default(f32)

fun f32 horner (f32 x) =
   let {c1,c2,c3,c4,c5} = {0.31938153,-0.356563782,1.781477937,-1.821255978,1.330274429}
   in x * (c1 + x * (c2 + x * (c3 + x * (c4 + x * c5))))

fun f32 fabs (f32 x) = if x < 0.0 then -x else x

fun f32 cnd0 (f32 d) =
   let k        = 1.0 / (1.0 + 0.2316419 * fabs(d))
   let p        = horner(k)
   let rsqrt2pi = 0.39894228040143267793994605993438
   in rsqrt2pi * exp32(-0.5*d*d) * p

fun f32 cnd (f32 d) =
   let c = cnd0(d)
   in if 0.0 < d then 1.0 - c else c

fun f32 go ({bool,f32,f32,f32} x) =
   let {call, price, strike, years} = x
   let r       = 0.08  -- riskfree
   let v       = 0.30  -- volatility
   let v_sqrtT = v * sqrt32(years)
   let d1      = (log32(price / strike) + (r + 0.5 * v * v) * years) / v_sqrtT
   let d2      = d1 - v_sqrtT
   let cndD1   = cnd(d1)
   let cndD2   = cnd(d2)
   let x_expRT = strike * exp32(-r * years)
   in if call then price * cndD1 - x_expRT * cndD2
              else x_expRT * (1.0 - cndD2) - price * (1.0 - cndD1)

fun [f32] blackscholes ([{bool,f32,f32,f32}] xs) =
   map (go, xs)

fun f32 main (int days) =
  let years = days // 365
  let a = map(+1, iota(days))
  let a = map(f32, a)
  let a = map(fn {bool,f32,f32,f32} (f32 x) => {int(x) % 2 == 0, 58.0 + 4.0 * x / f32(days), 65.0, x / 365.0}, a)
  in reduce(+, 0.0, blackscholes(a))
