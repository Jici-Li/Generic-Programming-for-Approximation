--------------------------------------------------------------------------------
--                       IntMultiplier_4x4_8_comb_uid5
-- VHDL generated for DummyFPGA @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Martin Kumm, Florent de Dinechin, Andreas Böttcher, Kinga Illyes, Bogdan Popa, Bogdan Pasca, 2012-
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X Y
-- Output signals: R
--  approx. input signal timings: X: 0.000000nsY: 0.000000ns
--  approx. output signal timings: R: 0.000000ns

library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
library std;
use std.textio.all;
library work;

entity IntMultiplier_4x4_8_comb_uid5 is
    port (X : in  std_logic_vector(3 downto 0);
          Y : in  std_logic_vector(3 downto 0);
          R : out  std_logic_vector(7 downto 0)   );
end entity;

architecture arch of IntMultiplier_4x4_8_comb_uid5 is
signal XX_m6 :  std_logic_vector(3 downto 0);
   -- timing of XX_m6: 0.000000ns
signal YY_m6 :  std_logic_vector(3 downto 0);
   -- timing of YY_m6: 0.000000ns
signal XX :  unsigned(-1+4 downto 0);
   -- timing of XX: 0.000000ns
signal YY :  unsigned(-1+4 downto 0);
   -- timing of YY: 0.000000ns
signal RR :  unsigned(-1+8 downto 0);
   -- timing of RR: 0.000000ns
begin
   XX_m6 <= X ;
   YY_m6 <= Y ;
   XX <= unsigned(X);
   YY <= unsigned(Y);
   RR <= XX*YY;
   R <= std_logic_vector(RR(7 downto 0));
end architecture;

--------------------------------------------------------------------------------
--                           IntAdder_10_comb_uid9
-- VHDL generated for DummyFPGA @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin (2008-2016)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X Y Cin
-- Output signals: R
--  approx. input signal timings: X: 2.100000nsY: 0.000000nsCin: 1.660000ns
--  approx. output signal timings: R: 3.190000ns

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity IntAdder_10_comb_uid9 is
    port (X : in  std_logic_vector(9 downto 0);
          Y : in  std_logic_vector(9 downto 0);
          Cin : in  std_logic;
          R : out  std_logic_vector(9 downto 0)   );
end entity;

architecture arch of IntAdder_10_comb_uid9 is
signal Rtmp :  std_logic_vector(9 downto 0);
   -- timing of Rtmp: 3.190000ns
begin
   Rtmp <= X + Y + Cin;
   R <= Rtmp;
end architecture;

--------------------------------------------------------------------------------
--                         FPMult_4_3_uid2_comb_uid3
-- VHDL generated for DummyFPGA @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Bogdan Pasca, Florent de Dinechin 2008-2021
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X Y
-- Output signals: R
--  approx. input signal timings: X: 0.000000nsY: 0.000000ns
--  approx. output signal timings: R: 3.190000ns

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity FPMult_4_3_uid2_comb_uid3 is
    port (X : in  std_logic_vector(4+3+2 downto 0);
          Y : in  std_logic_vector(4+3+2 downto 0);
          R : out  std_logic_vector(4+4+2 downto 0)   );
end entity;

architecture arch of FPMult_4_3_uid2_comb_uid3 is
   component IntMultiplier_4x4_8_comb_uid5 is
      port ( X : in  std_logic_vector(3 downto 0);
             Y : in  std_logic_vector(3 downto 0);
             R : out  std_logic_vector(7 downto 0)   );
   end component;

   component IntAdder_10_comb_uid9 is
      port ( X : in  std_logic_vector(9 downto 0);
             Y : in  std_logic_vector(9 downto 0);
             Cin : in  std_logic;
             R : out  std_logic_vector(9 downto 0)   );
   end component;

signal sign :  std_logic;
   -- timing of sign: 0.050000ns
signal expX :  std_logic_vector(3 downto 0);
   -- timing of expX: 0.000000ns
signal expY :  std_logic_vector(3 downto 0);
   -- timing of expY: 0.000000ns
signal expSumPreSub :  std_logic_vector(5 downto 0);
   -- timing of expSumPreSub: 1.050000ns
signal bias :  std_logic_vector(5 downto 0);
   -- timing of bias: 0.000000ns
signal expSum :  std_logic_vector(5 downto 0);
   -- timing of expSum: 2.100000ns
signal sigX :  std_logic_vector(3 downto 0);
   -- timing of sigX: 0.000000ns
signal sigY :  std_logic_vector(3 downto 0);
   -- timing of sigY: 0.000000ns
signal sigProd :  std_logic_vector(7 downto 0);
   -- timing of sigProd: 0.000000ns
signal excSel :  std_logic_vector(3 downto 0);
   -- timing of excSel: 0.000000ns
signal exc :  std_logic_vector(1 downto 0);
   -- timing of exc: 0.050000ns
signal norm :  std_logic;
   -- timing of norm: 0.000000ns
signal expPostNorm :  std_logic_vector(5 downto 0);
   -- timing of expPostNorm: 2.100000ns
signal sigProdExt :  std_logic_vector(7 downto 0);
   -- timing of sigProdExt: 0.550000ns
signal expSig :  std_logic_vector(9 downto 0);
   -- timing of expSig: 2.100000ns
signal sticky :  std_logic;
   -- timing of sticky: 0.550000ns
signal guard :  std_logic;
   -- timing of guard: 1.110000ns
signal round :  std_logic;
   -- timing of round: 1.660000ns
signal expSigPostRound :  std_logic_vector(9 downto 0);
   -- timing of expSigPostRound: 3.190000ns
signal excPostNorm :  std_logic_vector(1 downto 0);
   -- timing of excPostNorm: 3.190000ns
signal finalExc :  std_logic_vector(1 downto 0);
   -- timing of finalExc: 3.190000ns
begin
   sign <= X(7) xor Y(7);
   expX <= X(6 downto 3);
   expY <= Y(6 downto 3);
   expSumPreSub <= ("00" & expX) + ("00" & expY);
   bias <= CONV_STD_LOGIC_VECTOR(7,6);
   expSum <= expSumPreSub - bias;
   sigX <= "1" & X(2 downto 0);
   sigY <= "1" & Y(2 downto 0);
   SignificandMultiplication: IntMultiplier_4x4_8_comb_uid5
      port map ( X => sigX,
                 Y => sigY,
                 R => sigProd);
   excSel <= X(9 downto 8) & Y(9 downto 8);
   with excSel  select  
   exc <= "00" when  "0000" | "0001" | "0100", 
          "01" when "0101",
          "10" when "0110" | "1001" | "1010" ,
          "11" when others;
   norm <= sigProd(7);
   -- exponent update
   expPostNorm <= expSum + ("00000" & norm);
   -- significand normalization shift
   sigProdExt <= sigProd(6 downto 0) & "0" when norm='1' else
                         sigProd(5 downto 0) & "00";
   expSig <= expPostNorm & sigProdExt(7 downto 4);
   sticky <= sigProdExt(3);
   guard <= '0' when sigProdExt(2 downto 0)="000" else '1';
   round <= sticky and ( (guard and not(sigProdExt(4))) or (sigProdExt(4) ))  ;
   RoundingAdder: IntAdder_10_comb_uid9
      port map ( Cin => round,
                 X => expSig,
                 Y => "0000000000",
                 R => expSigPostRound);
   with expSigPostRound(9 downto 8)  select 
   excPostNorm <=  "01"  when  "00",
                               "10"             when "01", 
                               "00"             when "11"|"10",
                               "11"             when others;
   with exc  select  
   finalExc <= exc when  "11"|"10"|"00",
                       excPostNorm when others; 
   R <= finalExc & sign & expSigPostRound(7 downto 0);
end architecture;

