

create_project [lindex $argv 0] [lindex $argv 1]/[lindex $argv 0]/ -part xczu3eg-sbva484-1-e
set_property board_part em.avnet.com:ultra96:part0:1.0 [current_project]
create_bd_design "design_1"
update_compile_order -fileset sources_1
startgroup
create_bd_cell -type ip -vlnv xilinx.com:ip:zynq_ultra_ps_e:3.2 zynq_ultra_ps_e_0
endgroup
apply_bd_automation -rule xilinx.com:bd_rule:zynq_ultra_ps_e -config {apply_board_preset "1" }  [get_bd_cells zynq_ultra_ps_e_0]
set_property -dict [list CONFIG.PSU__USE__S_AXI_GP2 {1} CONFIG.PSU__USE__S_AXI_GP3 {1} CONFIG.PSU__CRL_APB__PL0_REF_CTRL__FREQMHZ {220}] [get_bd_cells zynq_ultra_ps_e_0]


set_property  ip_repo_paths [lindex $argv 2] [current_project]
update_ip_catalog
startgroup
create_bd_cell -type ip -vlnv xilinx.com:hls:SkyNet:1.0 SkyNet_0
endgroup
startgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/SkyNet_0/m_axi_INPUT_r} Slave {/zynq_ultra_ps_e_0/S_AXI_HP0_FPD} intc_ip {Auto} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP0_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/SkyNet_0/m_axi_OUTPUT_r} Slave {/zynq_ultra_ps_e_0/S_AXI_HP1_FPD} intc_ip {Auto} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/S_AXI_HP1_FPD]
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} Master {/zynq_ultra_ps_e_0/M_AXI_HPM0_FPD} Slave {/SkyNet_0/s_axi_AXILiteS} intc_ip {New AXI Interconnect} master_apm {0}}  [get_bd_intf_pins SkyNet_0/s_axi_AXILiteS]
endgroup
apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Clk_master {Auto} Clk_slave {/zynq_ultra_ps_e_0/pl_clk0 (214 MHz)} Clk_xbar {/zynq_ultra_ps_e_0/pl_clk0 (214 MHz)} Master {/zynq_ultra_ps_e_0/M_AXI_HPM1_FPD} Slave {/SkyNet_0/s_axi_AXILiteS} intc_ip {/ps8_0_axi_periph} master_apm {0}}  [get_bd_intf_pins zynq_ultra_ps_e_0/M_AXI_HPM1_FPD]
make_wrapper -files [get_files [lindex $argv 1]/[lindex $argv 0]/[lindex $argv 0].srcs/sources_1/bd/design_1/design_1.bd] -top
add_files -norecurse [lindex $argv 1]/[lindex $argv 0]/[lindex $argv 0].srcs/sources_1/bd/design_1/hdl/design_1_wrapper.v



launch_runs impl_1 -to_step write_bitstream -jobs 4
