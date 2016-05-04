fun (int,int) maxWithIndex((int,int) x, (int,int) y) =
  let (xv, xi) = x
  let (yv, yi) = y
  in if xv < yv then y
     else if yv < xv then x
                     else -- Prefer lowest index if the values are equal.
                       if xi < yi then x else y

fun int main([int,n] as) =
  let (_, i) = reduceComm(maxWithIndex, (0, 0), zip(as, iota(n)))
  in i
