fun i64 maxWithIndex(i64 x, i64 y) =
  let xv = int(x >> 32i64) & 0xFFFFFFFF
  let xi = int(x)
  let yv = int(y >> 32i64) & 0xFFFFFFFF
  let yi = int(y)
  in if xv < yv then y
     else if yv < xv then x
                     else -- Prefer lowest index if the values are equal.
                       if xi < yi then x else y

fun int main([int,n] as) =
  let i = reduceComm(maxWithIndex, 0i64,
                     map(|, zip(map(<<32i64, map(i64, as)), map(i64,iota(n)))))
  in int(i)
