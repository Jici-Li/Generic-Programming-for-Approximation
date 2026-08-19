--------------------------------------------------------------------------------
--                       InputIEEE_4_3_to_4_3_comb_uid2
-- VHDL generated for DummyFPGA @ 0MHz
-- This operator is part of the Infinite Virtual Library FloPoCoLib
-- All rights reserved 
-- Authors: Florent de Dinechin (2008)
--------------------------------------------------------------------------------
-- combinatorial
-- Clock period (ns): inf
-- Target frequency (MHz): 0
-- Input signals: X
-- Output signals: R
--  approx. input signal timings: X: 0.000000ns
--  approx. output signal timings: R: 0.000000ns

library ieee;
use ieee.std_logic_1164.all;
use ieee.std_logic_arith.all;
use ieee.std_logic_unsigned.all;
library std;
use std.textio.all;
library work;

entity InputIEEE_4_3_to_4_3_comb_uid2 is
    port (X : in  std_logic_vector(7 downto 0);
          R : out  std_logic_vector(4+3+2 downto 0)   );
end entity;

architecture arch of InputIEEE_4_3_to_4_3_comb_uid2 is
signal expX :  std_logic_vector(3 downto 0);
   -- timing of expX: 0.000000ns
signal fracX :  std_logic_vector(2 downto 0);
   -- timing of fracX: 0.000000ns
signal sX :  std_logic;
   -- timing of sX: 0.000000ns
signal expZero :  std_logic;
   -- timing of expZero: 0.000000ns
signal expInfty :  std_logic;
   -- timing of expInfty: 0.000000ns
signal fracZero :  std_logic;
   -- timing of fracZero: 0.000000ns
signal reprSubNormal :  std_logic;
   -- timing of reprSubNormal: 0.000000ns
signal sfracX :  std_logic_vector(2 downto 0);
   -- timing of sfracX: 0.000000ns
signal fracR :  std_logic_vector(2 downto 0);
   -- timing of fracR: 0.000000ns
signal expR :  std_logic_vector(3 downto 0);
   -- timing of expR: 0.000000ns
signal infinity :  std_logic;
   -- timing of infinity: 0.000000ns
signal zero :  std_logic;
   -- timing of zero: 0.000000ns
signal NaN :  std_logic;
   -- timing of NaN: 0.000000ns
signal exnR :  std_logic_vector(1 downto 0);
   -- timing of exnR: 0.000000ns
begin
   expX  <= X(6 downto 3);
   fracX  <= X(2 downto 0);
   sX  <= X(7);
   expZero  <= '1' when expX = (3 downto 0 => '0') else '0';
   expInfty  <= '1' when expX = (3 downto 0 => '1') else '0';
   fracZero <= '1' when fracX = (2 downto 0 => '0') else '0';
   reprSubNormal <= fracX(2);
   -- since we have one more exponent value than IEEE (field 0...0, value emin-1),
   -- we can represent subnormal numbers whose mantissa field begins with a 1
   sfracX <= fracX(1 downto 0) & '0' when (expZero='1' and reprSubNormal='1')    else fracX;
   fracR <= sfracX;
   -- copy exponent. This will be OK even for subnormals, zero and infty since in such cases the exn bits will prevail
   expR <= expX;
   infinity <= expInfty and fracZero;
   zero <= expZero and not reprSubNormal;
   NaN <= expInfty and not fracZero;
   exnR <= 
           "00" when zero='1' 
      else "10" when infinity='1' 
      else "11" when NaN='1' 
      else "01" ;  -- normal number
   R <= exnR & sX & expR & fracR; 
end architecture;

