<TeXmacs|2.1.1>

<style|<tuple|generic|british>>

<\body>
  <\hide-preamble>
    \;

    <assign||<macro|<with|font-series|bold|<compound|theorem-numbered|Assumption|<compound|the-theorem>>>>>

    <assign|theorem|<\macro|body>
      <surround|<compound|next-theorem>||<compound|render-theorem|<compound|theorem-numbered|<compound|theorem-text>|<compound|the-theorem>>|<arg|body>>>
    </macro>>

    <assign|proposition|<\macro|body>
      <surround|<compound|next-proposition>||<compound|render-theorem|<compound|proposition-numbered|Assumption|<compound|the-proposition>>|<arg|body>>>
    </macro>>
  </hide-preamble>

  <with|font-shape|right|<with|font-family|rm|<\with|font|palatino>
    Note 8.12<with|font|bookman|>

    <math|<around*|\<\|\|\>|\<cdummy\>|\<\|\|\>>\<assign\><around*|\<\|\|\>|\<cdummy\>|\<\|\|\>><rsub|2>
    norm \ >

    <section|Stochastic LM Algorithm<with|font|Palatino
    Linotype|font-series|medium|>>

    Consider the following least square problem\ 

    <\equation>
      min<rsub|x\<in\>R<rsup|d>> f<around*|(|x|)>\<assign\><frac|1|2><around*|\<\|\|\>|r<around*|(|x|)>|\<\|\|\>><rsup|2>=<frac|1|2><big|sum><rsub|i=1><rsup|M><around*|\||r<rsub|i><around*|(|x|)>|\|><rsup|2>
    </equation>

    where <math|r<rsub|i> > are continuously differentiable,
    <math|i=1,\<cdots\>,M>. Build a set <math|<with|font-shape|italic|\<cal-F\><rsub|b>>\<assign\><around*|{|F<rsub|l><around*|\|||\<nobracket\>>
    F<rsub|l>\<subset\><space|0.8spc>the<space|0.6spc>power <space|0.6spc>set
    <space|0.6spc>of<space|0.8spc><around*|{|1,\<cdots\>,d|}>,<infix-and><around*|\||F<rsub|l>|\|>=b|}>>,
    uniformly random choose <math|F\<in\>\<cal-F\><rsub|b>>

    <\equation>
      min f<around*|(|x<rsub|F<rsub|>>|)>\<assign\><frac|1|2><around*|\<\|\|\>|r<around*|(|x<rsub|F>|)>|\<\|\|\>><rsup|2>
    </equation>

    Construct a quadratic function as follows\ 

    <\equation>
      m<rsub|k><around*|(|x<rsub|F><rsup|k>+h|)>\<assign\>f<around*|(|x<rsup|k><rsub|F>|)>+g<rsup|T><around*|(|x<rsub|F><rsup|k>|)>*h+<frac|1|2>h<rsup|T>*H<around*|(|x<rsub|F><rsup|k>|)>*h+<frac|1|2>\<mu\><rsup|k><around*|\<\|\|\>|h|\<\|\|\>><rsup|2>
    </equation>

    where <math|\<mu\><rsub|k> >is the regularized parameter and
    <math|g<around*|(|x<rsub|F><rsup|k>|)>=><math|<frac|1|2><big|sum><rsub|i=1><rsup|M>\<nabla\><around*|\||r<rsub|i><around*|(|x<rsub|F>|)>|\|><rsup|2>=<big|sum><rsub|i=1><rsup|M>\<nabla\>r<rsub|i><around*|(|x<rsub|F>|)>*r<rsub|i><around*|(|x<rsub|F>|)>>,
    <math|<space|1.0spc>H<around*|(|x<rsub|F><rsup|k>|)>=><math|<big|sum><rsub|i=1><rsup|M>\<nabla\>r<rsub|i><around*|(|x<rsub|F>|)>\<nabla\>*r<rsub|i><around*|(|x<rsub|F>|)><rsup|T>>.
    the LM step is obtained by solving the subproblem: <text|<math|argmin
    <rsub|h> m<rsub|k><around*|(|x<rsub|F><rsup|k>+h|)>>>

    i.e <math|h<rsub|F><rsup|k>\<leftarrow\><around*|(|H<around*|(|x<rsub|F><rsup|k>|)>+\<mu\><rsup|k>I|)>h=-g<around*|(|x<rsub|F><rsup|k>|)>>.
    the ratio <math|\<rho\>> is defined as\ 

    <\equation>
      \<rho\><rsub|k>=<frac|f<around*|(|x<rsup|k><rsub|F>|)>-f<around*|(|x<rsup|k><rsub|F>+h<rsub|F><rsup|k>|)>|m<rsub|k><around*|(|x<rsup|k><rsub|F>|)>-m<rsub|k><around*|(|x<rsup|k><rsub|F>+h<rsub|F><rsup|k>|)>>
    </equation>

    \;

    <\with|par-mode|left>
      <\with|par-mode|left>
        <\named-algorithm>
          1.1<space|1em>Stochastic LM algorithm
        <|named-algorithm>
          Step 0 Random choose an parameter set <math|F> and
          <math|\<mu\>\<gtr\>0>, initial damping parameter
          <math|\<mu\><rsup|0>=\<mu\><around*|\<\|\|\>|r<around*|(|x<rsub|F><rsup|0>|)>|\<\|\|\>><rsup|2>>,
          constants <math|\<matheuler\>\<gtr\>1>, <math|\<mu\><rsub|min>> and
          <math|\<eta\><rsub|1>,\<eta\><rsub|2>\<gtr\>0>,set <math|k=0>

          \;

          Step 1 if a stopping criteria is satisfied, go to Step 0 or stop;
          otherwise, go to <space|>Step 2.\ 

          \;

          Step 2 obtain the direction <math|h<rsub|F><rsup|k>>

          \;

          Step 3 Compute the ratio <math|\<rho\><rsub|k> > in
          <math|<around*|(|4|)>>

          \;

          Step 4 if <math|\<rho\><rsub|k>\<geqslant\>\<eta\><rsub|1>> and
          <math|<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>\<geqslant\><frac|\<eta\><rsub|2>|\<mu\><rsup|k>>>,
          set <math|x<rsup|k+1><rsub|F>\<leftarrow\>x<rsup|k><rsub|F>+h<rsub|F><rsup|k>>
          and <math|\<mu\><rsup|k+1>=max<around*|(|\<mu\><rsup|k>/\<matheuler\>,\<mu\><rsub|min>|)>>;
          otherwise set <math|x<rsub|F><rsup|k+1>=x<rsub|F><rsup|k> and
          \<mu\><rsup|k+1>=\<matheuler\>\<mu\><rsup|k>*>. Then <math|k=k+1>,
          go to step 1.
        </named-algorithm>
      </with>
    </with>

    <section|Convergence analysis>

    <\proposition>
      Suppose <math|\<tau\><rsup|k>> is the solution of <math|argmin <rsub|h>
      m<rsub|k><around*|(|x<rsub|F><rsup|k>+h|)>>, then the following
      condition hold:\ 

      <\equation>
        m<rsub|k><around*|(|x<rsup|k><rsub|F>|)>-m<rsub|k><around*|(|x<rsup|k><rsub|F>+\<tau\><rsup|k>|)>\<geqslant\><frac|1|4>*<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>*min<around*|{|<frac|1|\<mu\><rsup|k>>,<frac|1|<around*|\<\|\|\>|H<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>>>|}>
      </equation>

      and\ 

      <\equation>
        <around*|\<\|\|\>|\<tau\><rsup|k>|\<\|\|\>>\<leqslant\><frac|2<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>>|\<mu\><rsup|k>>
      </equation>
    </proposition>

    <\proposition>
      Suppose <math|r<rsub|i><around*|(|x|)>> are continuously differentiable
      and <math|\<nabla\>*r<rsub|i><around*|(|x|)>> are Lipschitz continuous.
      <math|f<around*|(|x|)> >is bounded.
      <math|<around*|\<\|\|\>|H<around*|(|x|)>|\<\|\|\>>\<leqslant\>c > for a
      constant <math|c\<gtr\>0>

      <space|1em>Moreover, Under Assumption 2., there exist a constant
      Lipschitz coffecient <math|L\<gtr\>0> and constrain
      <math|x<rsub|F>,y<rsub|F>\<in\>F> <math|> .Then the descent lemma tells\ 

      <\equation>
        <around*|\||f<around*|(|y<rsub|F>|)>-f<around*|(|x<rsub|F>|)>-\<nabla\>f<around*|(|x|)><rsup|T>*<around*|(|y-x|)>|\|>\<leqslant\><frac|L|2><around*|\<\|\|\>|y-x|\<\|\|\>><rsup|2>
      </equation>
    </proposition>

    <\lemma>
      If Assumption 1. 2. holds. Then almost surely
      <math|\<mu\><rsup|k>\<gtr\>\<kappa\>> for any <math|\<kappa\>\<gtr\>0>.
    </lemma>

    <\proof>
      For a constant <math|\<kappa\>\<gtr\>0>. Prove by contradiction. Assume
      the set <math|<around*|{|k<around*|\|||\<nobracket\>>\<mu\><rsup|k>\<less\>\<kappa\>|}>>
      is infinite, also can conclude \ <math|P<around*|(|<around*|{|k<around*|\|||\<nobracket\>>\<mu\><rsup|k>\<less\>\<kappa\>|}>=\<infty\>|)>=\<alpha\>\<gtr\>0>.
      According to the algorithm, there has probability <math|a that
      \<mu\><rsup|k>> decrease infinite times. When the iteration is
      successful, <math|\<rho\><rsub|k>\<geqslant\>\<eta\><rsub|1>> and
      <math|<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>\<geqslant\><frac|\<eta\><rsub|2>|\<mu\><rsup|k>>>
      <space|0.6spc>holds. Consider the set<math|><space|1.0spc><math|S=<around*|{|k<around*|\||the
      k th iteration is successful|\<nobracket\>>|}>>\ 

      <\equation>
        <big|sum><rsub|k\<in\>S><around*|(|f<around*|(|x<rsup|k><rsub|F>|)>-f<around*|(|x<rsup|k><rsub|F>+h<rsub|F><rsup|k>|)>|)>\<geqslant\><big|sum><rsub|k\<in\>S><frac|\<eta\><rsub|1>|4>**<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>*min<around*|{|<frac|1|\<mu\><rsup|k>>,<frac|1|<around*|\<\|\|\>|H<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>>>|}>\<geqslant\><big|sum><rsub|k\<in\>S>*<frac|\<eta\><rsub|1>\<eta\><rsub|2>|4
        \<kappa\>>*min<around*|{|<frac|1|\<kappa\>>,<frac|1|c>|}>
      </equation>

      note that <math|<around*|\||S|\|>=\<infty\><space|0.6spc>happens with
      <space|0.4spc>positve<space|0.6spc> probability \<alpha\>>. So
      <math|E<around*|[|<big|sum><rsub|k\<in\>S><around*|(|f<around*|(|x<rsup|k><rsub|F>|)>-f<around*|(|x<rsup|k><rsub|F>+h<rsub|F><rsup|k>|)>|)>|]>=\<infty\>>.
      However, accoding to the assumption 2. <math|f<around*|(|x|)>> is
      bounded, which implies\ 

      <math|E<around*|[|<big|sum><rsub|k\<in\>S><around*|(|f<around*|(|x<rsup|k><rsub|F>|)>-f<around*|(|x<rsup|k><rsub|F>+h<rsub|F><rsup|k>|)>|)>|]>\<leqslant\>E<around*|[|2**f<around*|(|x<rsub|F>|)>|]>\<less\>\<infty\>>.
      We obtain the contradiction with <math|P<around*|(|<around*|{|k<around*|\|||\<nobracket\>>\<mu\><rsup|k>\<less\>\<kappa\>|}>=\<infty\>|)>=\<alpha\>\<gtr\>0>.
      the proof is completed.
    </proof>

    <\lemma>
      If Assumption 1. 2. holds. When <math|\<mu\><rsup|k>\<geqslant\>max<around*|{|c,<frac|8<around*|(|L+c|)>|1-\<eta\><rsub|1>>|}>>
      , then <math|\<rho\><rsub|k>\<gtr\>\<eta\><rsub|1>>.
    </lemma>

    <\proof>
      According to the Assumption 1. 2. and the condition
      <math|\<mu\><rsup|k>\<geqslant\>max<around*|{|c,<frac|8<around*|(|L+c|)>|1-\<eta\><rsub|1>>|}>>,
      we derive the inequality\ 

      <\equation>
        m<rsub|k><around*|(|x<rsup|k><rsub|F>|)>-m<rsub|k><around*|(|x<rsup|k><rsub|F>+h<rsub|F><rsup|k>|)>\<geqslant\><frac|1|4>**<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>**min<around*|{|<frac|1|\<mu\><rsup|k>>,<frac|1|<around*|\<\|\|\>|H<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>>>|}>
      </equation>

      recall the model <math|m<rsub|k><around*|(|x<rsup|k><rsub|F>|)>=><math|f<around*|(|x<rsup|k><rsub|F>|)>>.
      Rewrite the the descent lemma\ 

      <\equation*>
        f<around*|(|x<rsup|k><rsub|F>+h<rsub|F><rsup|k>|)>\<leqslant\>m<around*|(|x<rsup|k><rsub|F>|)>+g<around*|(|x<rsub|F><rsup|k>|)><rsup|T>**h<rsub|F><rsup|k>+<frac|L|2>*<around*|\<\|\|\>|h<rsub|F><rsup|k>|\<\|\|\>><rsup|2>
      </equation*>

      \;

      thus,\ 

      <\equation*>
        f<around*|(|x<rsup|k><rsub|F>+h<rsub|F><rsup|k>|)>-m<rsub|k><around*|(|x<rsup|k><rsub|F>+h<rsub|F><rsup|k>|)>\<leqslant\><frac|L|2>*<around*|\<\|\|\>|h<rsub|F><rsup|k>|\<\|\|\>><rsup|2>-<frac|1|2>*h<rsub|F><rsup|kT>*H<around*|(|x<rsub|F><rsup|k>|)>**h<rsub|F><rsup|k>-<frac|1|2>\<mu\><rsup|k>*<around*|\<\|\|\>|h<rsub|F><rsup|k>|\<\|\|\>><rsup|2>\<leqslant\><frac|L+c|2>*<around*|\<\|\|\>|h<rsub|F><rsup|k>|\<\|\|\>><rsup|2>\<leqslant\>2<around*|(|L+c|)>*<frac|<around*|\<\|\|\>|g<around*|(|x<rsup|k><rsub|F>|)>|\<\|\|\>>|\<mu\><rsup|k*<space|0.6spc>2>>
      </equation*>

      combined with the definition of <math|\<rho\><rsub|k>>,
      <math|1-\<rho\><rsub|k>\<leqslant\><frac|8<around*|(|L+c|)>|\<mu\><rsup|k>>\<Rightarrow\>\<rho\><rsub|k>\<geqslant\>1-<frac|8<around*|(|L+c|)>|\<mu\><rsup|k>>\<geqslant\>\<eta\><rsub|1>>.
      the proof is completed

      \;
    </proof>

    The following lemmas prove Algorithm 1.1 convergence with probability.
    Set <math|\<beta\>\<assign\>d*<sqrt|1-<frac|1|2\<mu\><rsup|k<space|0.6spc>
    2>*<around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|4>>>>

    <\lemma>
      if <math|b> in <math|\<cal-F\><rsub|b>> satisfies
      <math|b\<geqslant\>\<beta\>> , then the event\ 

      <\equation*>
        <math|I<rsub|k>\<assign\><around*|{|<around*|\||<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>-<frac|b|d><around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|2>|\|>\<less\><frac|1|\<mu\><rsup|k>>|}>>
      </equation*>

      \ has <math|P<around*|(|I<rsub|k>|)>\<gtr\><frac|1|2>>.
    </lemma>

    <\proof>
      \ Recall the definition <math|><math|<with|font-shape|italic|\<cal-F\><rsub|b>>\<assign\><around*|{|F<rsub|l><around*|\|||\<nobracket\>>
      F<rsub|l>\<subset\><space|0.8spc>the<space|0.6spc>power
      <space|0.6spc>set <space|0.6spc>of<space|0.8spc><around*|{|1,\<cdots\>,d|}>,<infix-and><around*|\||F<rsub|l>|\|>=b|}>>,
      so <math|<around*|\||\<cal-F\><rsub|b>|\|>=<binom|b|d>>, and <math|F>
      is uniformly random chosen from <math|\<cal-F\><rsub|b>>.\ 

      <\equation*>
        E<around*|[|<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>|]>=E<around*|[|<big|sum><rsub|i*\<in\>F><around*|(|g<rsub|i><around*|(|x<rsup|k>|)>|)><rsup|2>|]>=<big|sum><rsub|l><big|sum><rsub|i*\<in\>F<rsub|l>><around*|(|g<rsub|i><around*|(|x<rsup|k>|)>|)><rsup|2>*p<around*|(|F<rsub|l>|)>=<binom|b|d>*<frac|b|d>*<big|sum><rsub|i=0><rsup|d><around*|(|g<rsub|i><around*|(|x<rsup|k>|)>|)><rsup|2>
        <frac|1|<binom|b|d>>=<frac|b|d><around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|2>
      </equation*>

      and the variance\ 

      <\eqnarray*>
        <tformat|<table|<row|<cell|Var<around*|[|<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>|]>>|<cell|=>|<cell|E<around*|[|<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|4>|]>-E<around*|[|<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>|]><rsup|2>>>|<row|<cell|>|<cell|=>|<cell|<big|sum><rsub|l><around*|(|<big|sum><rsub|i*\<in\>F<rsub|l>><around*|(|g<rsub|i><around*|(|x<rsup|k>|)>|)><rsup|2>|)><rsup|2>*p<around*|(|F<rsub|l>|)>-<around*|(|<frac|b|d><around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|2>|)><rsup|2>>>|<row|<cell|>|<cell|\<leqslant\>>|<cell|<big|sum><rsub|l><around*|(|<big|sum><rsub|i*=0><rsup|d><around*|(|g<rsub|i><around*|(|x<rsup|k>|)>|)><rsup|2>|)><rsup|2>*p<around*|(|F<rsub|l>|)>-<frac|b<rsup|2>|d<rsup|2>><around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|4>>>|<row|<cell|>|<cell|=>|<cell|<around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|4>-<frac|b<rsup|2>|d<rsup|2>><around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|4>>>|<row|<cell|>|<cell|\<leqslant\>>|<cell|<around*|(|1-<frac|\<beta\><rsup|2>|d<rsup|2>>|)><around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|4>>>>>
      </eqnarray*>

      By the Chebyshev's inequality, we can obtain

      <\equation*>
        <math|P<around*|{|<around*|\||<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>-<frac|b|d><around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|2>|\|>\<less\><frac|1|\<mu\><rsup|k>>|}>\<gtr\>1-\<mu\><rsup|k>><rsup|<space|0.6spc>2>Var<around*|[|<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>|]>\<gtr\><frac|1|2>
      </equation*>

      \;
    </proof>

    <\theorem>
      Let the Assumption 1. 2. hold and condition in lemma 5 hold. Then the
      sequence of the total parameter <math|<around*|{|x<rsup|k>|}>>
      generated by Algorithm, almost surely satisfies\ 

      <\equation*>
        lim<rsub|k\<rightarrow\>\<infty\>>inf
        <around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>>=0
      </equation*>

      \;
    </theorem>

    <\proof>
      Prove this theorem by contradiction. Assume there exists
      <math|\<varepsilon\>\<gtr\>0> such that<space|2em> <math|
      <around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|2>\<geqslant\><frac|d|b>*\<varepsilon\>>
      for all <math|k\<geqslant\>k<rsub|0>>. According to the lemma 3.,
      \ there exists <math|k\<gtr\>k<rsub|1> > such that\ 

      <\equation>
        \<mu\><rsup|k>\<gtr\>\<chi\>\<assign\>max
        <around*|{|<frac|2|\<varepsilon\>>,<space|0.6spc><frac|2*\<eta\><rsub|2>|\<varepsilon\>>,<space|0.6spc>c,<frac|8<around*|(|L+c|)>|1-\<eta\><rsub|1>>,<space|0.6spc>\<gamma\>\<mu\><rsub|min>|}>
      </equation>

      Define <math|R<rsub|k>=log<rsub|\<gamma\>><around*|(|<frac|\<chi\>|\<mu\><rsup|k>>|)>,>
      by the assumption, <math|R<rsub|k>\<leqslant\>0 > for all
      <math|k\<gtr\>max<around*|(|k<rsub|0>,k<rsub|1>|)> >. \ 

      <space|1em>Since <math|\<mu\><rsup|k>\<gtr\>max<around*|{|c,<frac|8<around*|(|L+c|)>|1-\<eta\><rsub|1>>|}>>,
      then <math|\<rho\><rsub|k>\<geqslant\>\<eta\><rsub|1>>. So the
      iteration success just depends on <math|<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>>.
      In lemma 5., we have\ 

      <math|<around*|\||<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>-<frac|b|d><around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|2>|\|>\<less\><frac|1|\<mu\><rsup|k>>>
      with probability <math|\<upsilon\>\<gtr\><frac|1|2>>.
      <math|<around*|\||<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>-<frac|b|d><around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>><rsup|2>|\|>\<less\><frac|1|\<mu\><rsup|k>>\<less\><frac|\<varepsilon\>|2>>
      then <math|<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>\<geqslant\><frac|\<varepsilon\>|2>>.
      From <math|<around*|(|10|)> >, we can further obtain
      <math|<around*|\<\|\|\>|g<around*|(|x<rsub|F><rsup|k>|)>|\<\|\|\>><rsup|2>\<gtr\><frac|\<eta\><rsub|2>|\<mu\><rsup|k>>>
      which implies a successful iteration.\ 

      \ <math|E<around*|[|R<rsub|k+1>|]>=v<around*|(|log<rsub|\<gamma\>><around*|(|<frac|\<chi\>*\<gamma\>|\<mu\><rsup|k>*>|)>|)>+<around*|(|1-v|)>*log<rsub|\<gamma\>><around*|(|<frac|\<chi\>*|\<mu\><rsup|k>*\<gamma\>>|)>=><math|v<around*|(|log<rsub|\<gamma\>><around*|(|<frac|\<chi\>|\<mu\><rsup|k>>|)>+1|)>+<around*|(|1-v|)><around*|(|log<rsub|\<gamma\>><around*|(|<frac|\<chi\>|\<mu\><rsup|k>>|)>-1
      |)>\<geqslant\>R<rsub|k>>

      Since <math|<around*|\||R<rsub|k+1>-R<rsub|k>|\|>\<geqslant\>1>, we can
      conclude <math|P<around*|[|<space|0.6spc>lim<rsub|k\<rightarrow\>\<infty\>>
      sup \ R<rsub|k> <space|0.6spc>\<gtr\><space|0.6spc>0|]>=1> which leads
      to a contradiction to our assumption: <math|R<rsub|k>\<leqslant\>0 >
      for all <math|k\<gtr\>max<around*|(|k<rsub|0>,k<rsub|1>|)> .> So
      <math|lim<rsub|k\<rightarrow\>\<infty\>>inf
      <around*|\<\|\|\>|g<around*|(|x<rsup|k>|)>|\<\|\|\>>=0> holds \ almost
      surely.

      \ 
    </proof>
  </with>>>
</body>

<\initial>
  <\collection>
    <associate|font|FandolSong>
    <associate|font-base-size|11>
    <associate|font-family|rm>
    <associate|page-medium|paper>
    <associate|page-screen-margin|false>
    <associate|par-columns|1>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|1>>
    <associate|auto-2|<tuple|2|2>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|1<space|2spc>Stochastic
      LM Algorithm<with|font|<quote|Palatino
      Linotype>|font-series|<quote|medium>|>>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|2<space|2spc>Convergence
      analysis> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>