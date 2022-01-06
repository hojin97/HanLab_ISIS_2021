%%
%% $ ISIS.cls $
%%

%% 
\NeedsTeXFormat{LaTeX2e}[1995/12/01]
\ProvidesClass{ISIS}
              [2019/07/16 v0.01 LaTeX class for ISIS]
              
\usepackage{comment}
\@twosidetrue

\newif\if@cmfont
\@cmfontfalse
\newif\if@sloppy

\usepackage{courier}
\usepackage{indentfirst}
\usepackage{color}
\usepackage{colortbl}
\usepackage{xcolor}
\usepackage{array,tabularx}

\DeclareOption{cm}{\@cmfonttrue}
\DeclareOption{sloppy}{\@sloppytrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\ExecuteOptions{final}
\ProcessOptions
\RequirePackage[T1]{fontenc}
\RequirePackage{fancyhdr}
\RequirePackage{ifthen}
\RequirePackage{lastpage}
\RequirePackage{cite}
%\RequirePackage{picins}
\RequirePackage{tabu}
\RequirePackage{booktabs}
\RequirePackage{color}
\RequirePackage{float}%% for definition of foot* environment. NEVER REMOVE!!
%\RequirePackage{dblfloatfix}
\RequirePackage[fleqn]{amsmath}
\RequirePackage{amsfonts, amssymb, bm, mathrsfs}
\RequirePackage{multicol}
\PassOptionsToPackage{hyphens}{url}
\RequirePackage[colorlinks,hyperindex,breaklinks, hyperfootnotes=false]{hyperref}
\hypersetup{citecolor=blue, linkcolor=black, urlcolor=black}
\RequirePackage[justification=justified,singlelinecheck=false]{caption}

\if@cmfont
\AtBeginDocument{%
  \label{FirstPage}%% Mark the first page for checking the first page number
  }
\else
\@ifpackageloaded{mathptmx}{}{%
\def\rmdefault{ptm}
\DeclareSymbolFont{operators}   {OT1}{ztmcm}{m}{n}
\DeclareSymbolFont{letters}     {OML}{ztmcm}{m}{it}
\DeclareSymbolFont{symbols}     {OMS}{ztmcm}{m}{n}
\DeclareSymbolFont{largesymbols}{OMX}{ztmcm}{m}{n}
\DeclareSymbolFont{bold}        {OT1}{ptm}{bx}{n}
\DeclareSymbolFont{italic}      {OT1}{ptm}{m}{it}
\@ifundefined{mathbf}{}{\DeclareMathAlphabet{\mathbf}{OT1}{ptm}{bx}{n}}
\@ifundefined{mathit}{}{\DeclareMathAlphabet{\mathit}{OT1}{ptm}{m}{it}}
\DeclareMathAlphabet{\mathcal}{OMS}{cmsy}{m}{n}
\DeclareMathSymbol{\omicron}{0}{operators}{`\o}
\renewcommand{\jmath}{%
  \PackageError{IJCAS}
  {The symbols \protect\jmath, \protect\amalg\space and
  \protect\coprod\MessageBreak
  are not available with this package along with mathptmx}
  {Type \space <return> \space to proceed;
  your command will be ignored.}}
\let\amalg=\jmath
\let\coprod=\jmath
\thinmuskip=2mu
\medmuskip=2.5mu plus 1mu minus 1mu
\thickmuskip=4mu plus 1.5mu minus 1mu
\DeclareRobustCommand\hbar{{%
 \dimen@.03em%
 \dimen@ii.06em%
 \def\@tempa##1##2{{%
   \lower##1\dimen@\rlap{\kern##1\dimen@ii\the##2 0\char22}}}%
 \mathchoice\@tempa\@ne\textfont
            \@tempa\@ne\textfont
            \@tempa\defaultscriptratio\scriptfont
            \@tempa\defaultscriptscriptratio\scriptscriptfont
  h}}
\def\boldmath{%
   \PackageWarning%
   {mathptmx}%
   {There are no bold math fonts}%
   \global\let\boldmath=\relax
}

\AtBeginDocument{%
  \label{FirstPage}%% Mark the first page for checking the first page number
  \DeclareFontFamily{U}{msa}{}%
  \DeclareFontShape{U}{msa}{m}{n}{<->msam10}{}%
  \DeclareFontFamily{U}{msb}{}%
  \DeclareFontShape{U}{msb}{m}{n}{<->msbm10}{}%
  \DeclareFontFamily{U}{euf}{}%
  \DeclareFontShape{U}{euf}{m}{n}{<-6>eufm5<6-8>eufm7<8->eufm10}{}%
  \DeclareFontShape{U}{euf}{b}{n}{<-6>eufb5<6-8>eufb7<8->eufb10}{}%
  \@ifpackageloaded{amsfonts}{\let\hbar\s@vedhbar}{}
  \@ifpackageloaded{amsmath}{}{%
    \newdimen\big@size
    \addto@hook\every@math@size{\setbox\z@\vbox{\hbox{$($}\kern\z@}%
       \global\big@size 1.2\ht\z@}
    \def\bBigg@#1#2{%
       {\hbox{$\left#2\vcenter to#1\big@size{}\right.\n@space$}}}
    \def\big{\bBigg@\@ne}
    \def\Big{\bBigg@{1.5}}
    \def\bigg{\bBigg@\tw@}
    \def\Bigg{\bBigg@{2.5}}
  }%
 }%
}
\fi

\def\@vpt{5}
\def\@vipt{6}
\def\@viipt{7}
\def\@viiipt{8}
\def\@ninesize{9}
\def\@ixpt{9}
\def\@xpt{10}% 10.5 -> 10 on 2011
\def\@elevensize{10.0}% 11.0 -> 10.0 on 2011
\def\@xipt{11.25}
\def\@xiipt{12.24}
\def\@xivpt{15}
\def\@xviipt{17.28}
\def\@xxpt{20.74}
\def\@xxvpt{24.88}
\renewcommand\normalsize{%
   \@setfontsize\normalsize\@xpt\@xiipt
   \abovedisplayskip 10\p@ \@plus2\p@ \@minus5\p@
   \abovedisplayshortskip \z@ \@plus3\p@
   \belowdisplayshortskip 6\p@ \@plus3\p@ \@minus3\p@
   \belowdisplayskip \abovedisplayskip
   \let\@listi\@listI}
\normalsize
\newcommand\small{%
   \@setfontsize\small\@ixpt{11}%
   \abovedisplayskip 8.5\p@ \@plus3\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus2\p@
   \belowdisplayshortskip 4\p@ \@plus2\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 4\p@ \@plus2\p@ \@minus2\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\newcommand\footnotesize{%
%   \@setfontsize\footnotesize\@ninesize{9.5}%
   \@setfontsize\footnotesize\@ninesize{9}%
   \abovedisplayskip 6\p@ \@plus2\p@ \@minus4\p@
   \abovedisplayshortskip \z@ \@plus\p@
   \belowdisplayshortskip 3\p@ \@plus\p@ \@minus2\p@
   \def\@listi{\leftmargin\leftmargini
               \topsep 3\p@ \@plus\p@ \@minus\p@
               \parsep 2\p@ \@plus\p@ \@minus\p@
               \itemsep \parsep}%
   \belowdisplayskip \abovedisplayskip
}
\newcommand\scriptsize{\@setfontsize\scriptsize\@viipt\@viiipt}
\newcommand\tiny{\@setfontsize\tiny\@vpt\@vipt}
\newcommand\ninesize{\@setfontsize\ninesize\@ninesize{9.5}}
\newcommand\elevensize{\@setfontsize\elevensize\@elevensize{11.5}}
\newcommand\large{\@setfontsize\large\@xiipt{14}}
\newcommand\Large{\@setfontsize\Large\@xivpt{18}}
\newcommand\LARGE{\@setfontsize\LARGE\@xviipt{22}}
\newcommand\huge{\@setfontsize\huge\@xxpt{25}}
\newcommand\Huge{\@setfontsize\Huge\@xxvpt{30}}
\setlength\parindent{1em}
\setlength\smallskipamount{3\p@ \@plus 1\p@ \@minus 1\p@}
\setlength\medskipamount{6\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\bigskipamount{12\p@ \@plus 4\p@ \@minus 4\p@}
\setlength\headheight{12\p@}
\setlength\headsep   {25\p@}
\setlength\topskip   {10\p@}
\setlength\footskip{30\p@}
\if@compatibility \setlength\maxdepth{4\p@} \else
\setlength\maxdepth{.5\topskip} \fi
%%%%% *
\setlength{\skip\footins}{9\p@ \@plus 4\p@ \@minus 2\p@}
\setlength\floatsep    {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\textfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
\setlength\intextsep   {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\dblfloatsep    {12\p@ \@plus 2\p@ \@minus 2\p@}
\setlength\dbltextfloatsep{20\p@ \@plus 2\p@ \@minus 4\p@}
\setlength\@fptop{0\p@ \@plus 1fil}
\setlength\@fpsep{8\p@ \@plus 2fil}
\setlength\@fpbot{0\p@ \@plus 1fil}
\setlength\@dblfptop{0\p@ \@plus 1fil}
\setlength\@dblfpsep{8\p@ \@plus 2fil}
\setlength\@dblfpbot{0\p@ \@plus 1fil}
\setlength\partopsep{2\p@ \@plus 1\p@ \@minus 1\p@}
\def\@listi{\leftmargin\leftmargini
            \parsep 2\p@ \@plus2\p@ \@minus\p@
            \topsep 6\p@ \@plus2\p@ \@minus4\p@
            \itemsep0\p@ \@plus1\p@ \@minus\p@}
\let\@listI\@listi
\@listi
\def\@listii {\leftmargin\leftmarginii
              \labelwidth\leftmarginii
              \advance\labelwidth-\labelsep
              \topsep    4\p@ \@plus2\p@ \@minus\p@
              \parsep    2\p@ \@plus\p@  \@minus\p@
              \itemsep   0\p@}
\def\@listiii{\leftmargin\leftmarginiii
              \labelwidth\leftmarginiii
              \advance\labelwidth-\labelsep
              \topsep    2\p@ \@plus\p@\@minus\p@
              \parsep    \z@
              \partopsep \p@ \@plus\z@ \@minus\p@
              \itemsep   0\p@}
\def\@listiv {\leftmargin\leftmarginiv
              \labelwidth\leftmarginiv
              \advance\labelwidth-\labelsep}
\def\@listv  {\leftmargin\leftmarginv
              \labelwidth\leftmarginv
              \advance\labelwidth-\labelsep}
\def\@listvi {\leftmargin\leftmarginvi
              \labelwidth\leftmarginvi
              \advance\labelwidth-\labelsep}
%%%
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.75}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.8}
\renewcommand\dblfloatpagefraction{.75}

\newcommand\maketitle{\par
  \begingroup
	\renewcommand\thefootnote{}%
    \def\@makefnmark{\rlap{\@textsuperscript{\normalfont\@thefnmark}}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hb@xt@1.5em{%
                \hss\@textsuperscript{\normalfont\@thefnmark}}##1}%
      \ifnum \col@number=\@ne
        \@maketitle
      \else
   \twocolumn[\@maketitle]%
      \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \global\let\thanks\relax
  \global\let\maketitle\relax
  \global\let\@maketitle\relax
  \global\let\@thanks\@empty
  \global\let\@author\@empty
  \global\let\@date\@empty
  \global\let\@title\@empty
  \global\let\title\relax
  \global\let\author\relax
  \global\let\date\relax
  \global\let\and\relax
}

\def\@maketitle{%
  \newpage
  \null
  \vskip 1.0em%
  \begin{center}%
    {\minipage{\textwidth}{\fontsize{14pt}{16pt}\selectfont{{\bf\@title}} \par}\endminipage}%
    \vskip .5em%
    {%
    \vskip 11mm
      \minipage{\textwidth}{
      \begin{center}
        {\@author}%
 	   \end{center}
      }%
      \endminipage}
      \vskip 4.5mm
      {\fontsize{9pt}{11pt}\selectfont\institutename\par}
      {\fontsize{9pt}{11pt}\selectfont\texttt{\@email}}
  \vskip 12mm
  \abstract@produce
  \vskip 4.5mm
  \keyword@produce
  \vskip -1pc%\vskip -0.5 em%
  \end{center}%
	\vskip 1pc
}
%
%%%
\newbox\abstractbox
\newenvironment{abstract}{%
   \lrbox\abstractbox
     \minipage{102mm}%
     \bfseries\normalsize\abstractname~%
	 \normalfont\fontsize{9pt}{12pt}\selectfont\ignorespaces%
  }{%
   \endminipage\endlrbox}
\def\abstract@produce{%
   \leavevmode\box\abstractbox\par
}
%%%
\newbox\keywordbox
\newenvironment{keywords}{%
   \lrbox\keywordbox
     \minipage{102mm}%
     \bfseries\normalsize\keywordname~%
	 \normalfont\normalsize\ignorespaces%
  }{%
   \endminipage\endlrbox}
\def\keyword@produce{%
   \leavevmode\box\keywordbox\par
}
%%%
\newbox\footnotebox
\newenvironment{makefoot}{%
   \lrbox\footnotebox
     \minipage{.95\textwidth}%
     \normalfont\fontsize{8.5pt}{8.5pt}\ignorespaces%
  }{%
   \endminipage\endlrbox}
\def\footnote@produce{%
   \leavevmode\box\footnotebox\par
}

\newcommand{\email}[1]{
\gdef\@email{#1}}

%%% SECTION commands:
%%% \section and \subsection are effective, but other commands will be preserved
%%% for compatibility.
\newcommand*{\@seccntformat@section}[1]{\csname the#1\endcsname\quad}
\newcommand*{\@seccntformat@subsection}[1]{\csname the#1\endcsname\quad}
\newcommand*{\@seccntformat@subsubsection}[1]{\csname the#1\endcsname\quad}
\let\@@seccntformat\@seccntformat
\renewcommand*{\@seccntformat}[1]{%
  \expandafter\ifx\csname @seccntformat@#1\endcsname\relax
    \expandafter\@@seccntformat
  \else
    \expandafter
      \csname @seccntformat@#1\expandafter\endcsname
  \fi
    {#1}%
}



\setcounter {secnumdepth}{5}
\newcounter {section}
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand\thesection {\@arabic\c@section}
\renewcommand\thesubsection   {\thesection.\@arabic\c@subsection}
\renewcommand\thesubsubsection{\thesubsection.\@arabic\c@subsubsection}
\renewcommand\theparagraph    {\thesubsubsection.\@arabic\c@paragraph}
\renewcommand\thesubparagraph {\theparagraph.\@arabic\c@subparagraph}


\newcommand\section{\@startsection {section}{1}{\z@}%
                   {-3.5ex \@plus -.5ex \@minus -.2ex}%
                   {2.0ex \@plus.2ex}%
                   {\noindent\normalfont\bfseries\fontsize{12pt}{12pt}\selectfont}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                      {-2.8ex\@plus -.5ex \@minus -.2ex}%
                      {0.5ex \@plus .1ex}%
                      {\noindent\normalfont\bfseries\fontsize{10pt}{10pt}\selectfont}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                         {-2.8ex\@plus -.5ex \@minus -.2ex}%
                         {0.5ex \@plus .1ex}%
                         {\noindent\normalfont\bfseries\fontsize{10pt}{10pt}\selectfont}}

\leftmargin  \leftmargini
\setlength\leftmargini {1.8em}
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\setlength\leftmarginv  {.5em}
\setlength\leftmarginvi {.5em}
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\@arabic\c@enumi}
\renewcommand\theenumii{\@alph\c@enumii}
\renewcommand\theenumiii{\@roman\c@enumiii}
\renewcommand\theenumiv{\@Alph\c@enumiv}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{\textbullet}
\newcommand\labelitemii{\normalfont\bfseries \textendash}
\newcommand\labelitemiii{\textasteriskcentered}
\newcommand\labelitemiv{\textperiodcentered}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand*\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item\relax}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item\relax}
               {\endlist}
\newcounter{appendixcnt}
\setcounter{appendixcnt}{0}
\newcommand\appendix{\par\medskip
  \stepcounter{appendixcnt}%
  \section*{\MakeUppercase\appendixname~\Alph{appendixcnt}}\par
  \setcounter{section}{0}%
  \setcounter{subsection}{0}%
  \setcounter{equation}{0}%
  \def\theequation{\Alph{appendixcnt}.\arabic{equation}}%
  \gdef\thesubsection{\Alph{appendixcnt}.\arabic{subsection}}%
}

\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\renewcommand \theequation {\@arabic\c@equation}
\newcounter{figure}
\renewcommand \thefigure {\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename\nobreakspace{\thefigure}}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}
\renewcommand\thetable{\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename\nobreakspace{\thetable}}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{5\p@}

\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\DeclareRobustCommand*\cal{\@fontswitch\relax\mathcal}
\DeclareRobustCommand*\mit{\@fontswitch\relax\mathnormal}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg{2.55em}
\newcommand\@dotsep{4.5}
\newdimen\bibindent
\setlength\bibindent{1.5em}
\renewcommand{\@biblabel}[1]{#1.}
\newenvironment{thebibliography}[1]
     {%
      \subsection*{\indent\refname}%
      \vskip 0.5pc
      \fontsize{9pt}{11pt}\selectfont
  %    \@mkboth{\refname}{\refname}%
      \list{\@biblabel{\@arabic\c@enumiv}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \setlength\itemsep{\parsep}
            \@openbib@code
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\@arabic\c@enumiv}}%
      \sloppy
      \clubpenalty4000
      \@clubpenalty \clubpenalty
      \widowpenalty4000%
      \sfcode`\.\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{\hskip .11em\@plus.33em\@minus.07em}
\let\@openbib@code\@empty
\newcommand\@idxitem{\par\hangindent 40\p@}
\newcommand\subitem{\@idxitem \hspace*{20\p@}}
\newcommand\subsubitem{\@idxitem \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule\@width.4\columnwidth
  \kern2.6\p@}
\newcommand\@makefntext[1]{%
    \parindent 1em%
    \noindent
    \hb@xt@1.8em{\hss\@makefnmark}#1}
\newcommand\refname{{References}}
\newcommand\figurename{{Fig.}} 
\newcommand\tablename{{Table}}  
\newcommand\appendixname{Appendix}
\newcommand\abstractname{{Abstract.}}
\newcommand\keywordname{{Keywords:}}
\def\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{12\p@}
\setlength\columnseprule{0\p@}
\pagestyle{plain}
\pagenumbering{arabic}
\raggedbottom
\onecolumn
\if@sloppy
   \sloppy
\else
\fi
\flushbottom
\oddsidemargin=-15pt
\evensidemargin\oddsidemargin
\headheight=12pt

\marginparsep=10pt

\hoffset=23mm%0pt
\voffset=0pt

\topmargin=-16mm%-52pt
\headsep=31.5mm%15pt%25pt
\footskip=52.5mm%45mm%30pt

\textwidth=122mm%483pt
\textheight=195mm%672pt%682pt

\marginparwidth=48pt
\marginparpush=5pt

\paperheight=297mm%796pt
\paperwidth=210mm%597pt

\columnsep=2.14em % added on 2011



\RequirePackage{amsthm,amssymb}
\thm@style{plain}
\newtoks\thm@bodyfont  \thm@bodyfont{\normalfont}
\newtoks\thm@headfont  \thm@headfont{\bfseries}
\newtoks\thm@notefont  \thm@notefont{}
\newtoks\thm@headpunct \thm@headpunct{.}
\newskip\thm@preskip \newskip\thm@postskip
\def\thm@space@setup{%
  \thm@preskip=\topsep \thm@postskip=\thm@preskip
} 
\def\th@plain{%
  \let\thm@indent\indent % no indent
  \thm@headpunct{:}% add period after heading
  \normalfont% body font
} 
\def\@endtheorem{\endtrivlist\@endpefalse} 

%%% borrowed from amsart
\def\qed{\ifhmode\unskip\nobreak\fi\hfill
  \ifmmode\square\else$\m@th\square$\fi}

\renewenvironment{proof}[1][\proofname]{\par
  \normalfont
  \topsep6\p@\@plus6\p@ \trivlist
  \item[\hskip\labelsep\indent
  \bfseries
    #1\@addpunct{:}]\ignorespaces
}{%
  \qed\endtrivlist
}

%%% BIOGRAPHY : borrowed from IEEEtran and modified
\newif\if@biographyTOCentrynotmade
\global\@biographyTOCentrynotmadetrue

\newdimen\@tmpdimenA
\newdimen\@tmpdimenB 
\newcount\@tmpcountA
\newcount\@tmpcountB 
\begingroup
\catcode`\Q=3
\long\gdef\@ifmtarg#1{\@xifmtarg#1QQ\@secondoftwo\@firstoftwo\@nil}
\long\gdef\@xifmtarg#1#2Q#3#4#5\@nil{#4}
\endgroup 
\def\@needspace#1#2{\penalty-100\begingroup%shield temp variable
\@tmpdimenB\pagegoal\advance\@tmpdimenB-\pagetotal% space left
\ifdim #1>\@tmpdimenB\relax% not enough space left
\ifdim\@tmpdimenB>\z@\relax #2\fi%
\newpage%
\fi\endgroup} 

\newcounter{biography}
\setcounter{biography}{0}

\def\@photowidth{25mm}   % width of the biography photo area
\def\@photodepth{30mm}   % depth (height) of the biography photo area
\def\@hangwidth{27.5mm}  % width cleared for the biography photo area
\def\@hangdepth{31.5mm}  % depth cleared for the biography photo area
\def\@skipN{1.5\baselineskip}% nominal value of the vskip above the biography

\RequirePackage{graphicx}

\newcommand\biography[3]%
{\def\temp{#1}\ifx\temp\empty
{\noindent\fontsize{9pt}{10pt}\selectfont{\textbf{#2}}}
 {\fontsize{9pt}{10pt}\selectfont{#3}}
\else
\begin{@biography}[{\includegraphics[width=25mm,height=30mm,clip,keepaspectratio]{#1}}]{\fontsize{9pt}{10pt}\selectfont{\textbf{#2}}}
 {\fontsize{9pt}{10pt}\selectfont{#3}}
 \end{@biography}
\fi
}

\newenvironment{@biography}[2][]{\normalfont\small%
\unitlength 1in\parskip=0pt\par\parindent 1em\interlinepenalty500%
\@tmpdimenA=\@hangdepth%
\advance\@tmpdimenA by \@skipN%
\advance\@tmpdimenA by 1\baselineskip%
\@needspace{\@tmpdimenA}{\relax}%
\vskip \@skipN plus 1fil minus 0\baselineskip%
\def\@tempbiographybox{{\setlength{\fboxsep}{0pt}\framebox{\begin{minipage}[b][\@photodepth][c]{\@photowidth}\centering PLACE\\ PHOTO\\ HERE \end{minipage}}}}%
\@ifmtarg{#1}{\relax}{\def\@tempbiographybox{\mbox{\begin{minipage}[b][\@photodepth][c]{\@photowidth}%
\centering%
#1%
\end{minipage}}}}% end if optional argument supplied
\if@biographyTOCentrynotmade%
\setcounter{biography}{-1}%
\refstepcounter{biography}%
\addcontentsline{toc}{section}{Biographies}%
\global\@biographyTOCentrynotmadefalse%
\fi%
\refstepcounter{biography}%
\addcontentsline{toc}{subsection}{#2}%
\let\@ORGparCMD=\par% 
\edef\par{\hfil\break\indent}%
\settoheight{\@tmpdimenA}{\@tempbiographybox}%
\@tmpdimenB=\@hangdepth%
\@tmpcountA=\@tmpdimenB% countA has the hang depth
\divide\@tmpcountA by \baselineskip%
\advance\@tmpcountA by 1%
\hangindent\@hangwidth%
\hangafter-\@tmpcountA%
\settoheight{\@tmpdimenB}{\mbox{T}}%
\noindent\makebox[0pt][l]{\hspace{-\@hangwidth}\raisebox{\@tmpdimenB}[0pt][0pt]{\raisebox{-\@photodepth}[0pt][0pt]{\@tempbiographybox}}}%
\noindent\textbf{#2\ }\ignorespaces}
{
\relax\let\par=\@ORGparCMD\par%
 \ifnum \prevgraf <\@tmpcountA\relax%
    \advance\@tmpcountA by -\prevgraf%
    \advance\@tmpcountA by -1\relax%
    \@tmpdimenA=\baselineskip%
    \multiply\@tmpdimenA by \@tmpcountA%
    \noindent\rule{0pt}{\@tmpdimenA}%
\fi%
\par\normalfont
}
%\end{comment}

\let\lastpagebalance\balance

\newcommand\conftitle[1]%
	{\gdef\journal@title{#1}}
\newcommand\confyear[1]%
	{\gdef\journal@year{#1}}
\newcommand\confnumber[1]%
	{\gdef\journal@number{#1}}

\conftitle{}
\confyear{}
\confnumber{}

\newcommand\setarticlestartpagenumber[1]%
	{
	\setcounter{page}{#1}
	\gdef\startingpage{#1}
	}

\newenvironment{reference}
	{\begin{thebibliography}{00}}
	{\end{thebibliography}}


\newcounter{@inst}
\newcounter{@auth}

\def\institute#1{\gdef\@institute{#1}}

\def\institutename{\par
 \begingroup
 \parskip=\z@
 \parindent=\z@
 \setcounter{@inst}{1}%
 \def\and{\par\stepcounter{@inst}%
 \noindent$^{\the@inst}$\ignorespaces}%
 \setbox0=\vbox{\def\thanks##1{}\@institute}%
 \ifnum\c@@inst=1\relax
 \else
   \setcounter{footnote}{\c@@inst}%
   \setcounter{@inst}{1}%
   \noindent$^{\the@inst}$\ignorespaces
 \fi
 \ignorespaces
 \@institute\par
 \endgroup}

\def\@thanks{}

\def\@fnsymbol#1{\ifcase#1%
   \or \dagger\or \ddagger\or
   \mathchar "278\or \mathchar "27B\or \|\or **\or \dagger\dagger
   \or \ddagger\ddagger\else\@ctrerr\fi\relax}

\def\inst#1{\unskip$^{#1}$}

\usepackage{microtype}

\pagestyle{fancy}
\fancyhf{}
\fancyhead[C]{\fontsize{11pt}{11pt}\selectfont\textbf{\textsf{ISIS\journal@year}} \begin{minipage}[t]{0.85\linewidth}
\fontsize{11pt}{11pt}\selectfont
\sf \textls[-50]{The \journal@number\ International Symposium on Advanced Intelligent Systems 
%and \journal@year~International Conference on Biometrics and Kansei Engineering
}\end{minipage}} 
\fancyfoot[C]{\vspace{-2pc}\fontsize{12pt}{12pt}\selectfont\textbf{\textsf{\thepage}}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}

\thispagestyle{plain}
\fancypagestyle{plain}{
\fancyhf{}
\fancyhead[C]{\fontsize{11pt}{11pt}\selectfont\textbf{\textsf{ISIS\journal@year}} \begin{minipage}[t]{0.85\linewidth}
\fontsize{11pt}{11pt}\selectfont
\sf \textls[-50]{The \journal@number\ International Symposium on Advanced Intelligent Systems 
%and \journal@year~International Conference on Biometrics and Kansei Engineering
}\end{minipage}} 
\fancyfoot[C]{\vspace{-1.15pc}\fontsize{12pt}{12pt}\selectfont\textbf{\textsf{\thepage}}}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
}

\captionsetup{justification=centering, labelsep=period, font=small}