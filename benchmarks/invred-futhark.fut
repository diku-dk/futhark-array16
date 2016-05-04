fun (i8, i8, i8, i8) unpackInt(int x) =
  (i8(x >>> 24), i8(x >>> 16), i8(x >>> 8), i8(x >>> 0))

fun int packInt(i8 a, i8 b, i8 c, i8 d) =
  ((i32(a)&0xFF) << 24) |
  ((i32(b)&0xFF) << 16) |
  ((i32(c)&0xFF) << 8)  |
  ((i32(d)&0xFF) << 0)

fun int twoByTwoMult(int x, int y) =
  let (x11, x12,
       x21, x22) = unpackInt(x)
  let (y11, y12,
       y21, y22) = unpackInt(y)

  let z11 = x11 * y11 + x12 * y21
  let z12 = x11 * y12 + x12 * y22
  let z21 = x21 * y11 + x22 * y21
  let z22 = x21 * y12 + x22 * y22
  in packInt(z11, z12,
             z21, z22)

fun int eachAdd(int x, int y) =
  let x' = i8(x)
  let (y11, y12,
       y21, y22) = unpackInt(y)
  in packInt(y11+x', y12+x',
             y21+x', y22+x')

fun int main([int] a) =
  loop(s = 1) = for i < 42 do
    let a' = map(+s, a)
    in reduce(twoByTwoMult, 0xFF00FF00, a')
  in s
