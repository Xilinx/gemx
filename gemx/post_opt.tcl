set_property DONT_TOUCH TRUE [get_cells CL/xcl_design_i/expanded_region/u_ocl_region/dr_i/gemxKernel_*]
opt_design -control_set_merge -hier_fanout_limit 512 -sweep
set_property DONT_TOUCH FALSE [get_cells CL/xcl_design_i/expanded_region/u_ocl_region/dr_i/gemxKernel_*]
create_pblock pblock_0
resize_pblock pblock_0 -add CLOCKREGION_X2Y13:CLOCKREGION_X4Y14
create_pblock pblock_1
resize_pblock pblock_1 -add CLOCKREGION_X0Y5:CLOCKREGION_X1Y9
create_pblock pblock_2
resize_pblock pblock_2 -add CLOCKREGION_X2Y0:CLOCKREGION_X3Y2
create_pblock pblock_3
resize_pblock pblock_3 -add CLOCKREGION_X2Y10:CLOCKREGION_X3Y12
add_cells_to_pblock pblock_0 [get_cells [list CL/xcl_design_i/expanded_region/u_ocl_region/dr_i/gemxKernel_0/inst/kernelOpLow_U0/grp_runGemm_fu_*/grp_GemmBlocks_fu_*/GemmWrite_U0]] -clear_locs
add_cells_to_pblock pblock_1 [get_cells [list CL/xcl_design_i/expanded_region/u_ocl_region/dr_i/gemxKernel_1/inst/kernelOpLow_U0/grp_runGemm_fu_*/grp_GemmBlocks_fu_*/GemmWrite_U0]] -clear_locs
add_cells_to_pblock pblock_2 [get_cells [list CL/xcl_design_i/expanded_region/u_ocl_region/dr_i/gemxKernel_2/inst/kernelOpLow_U0/grp_runGemm_fu_*/grp_GemmBlocks_fu_*/GemmWrite_U0]] -clear_locs
add_cells_to_pblock pblock_3 [get_cells [list CL/xcl_design_i/expanded_region/u_ocl_region/dr_i/gemxKernel_3/inst/kernelOpLow_U0/grp_runGemm_fu_*/grp_GemmBlocks_fu_*/GemmWrite_U0]] -clear_locs
resize_pblock pblock_CL_top -add CLOCKREGION_X0Y10:CLOCKREGION_X5Y14 -locs keep_all
resize_pblock pblock_CL_mid -add CLOCKREGION_X0Y5:CLOCKREGION_X1Y9 -locs keep_all
resize_pblock pblock_CL_bot -add CLOCKREGION_X0Y0:CLOCKREGION_X3Y4 -locs keep_all
add_cells_to_pblock pblock_CL_bot [get_cells [list CL/xcl_design_i/expanded_region/u_ocl_region/dr_i/gemxKernel_2]] -clear_locs
add_cells_to_pblock pblock_2 [get_cells [list CL/xcl_design_i/expanded_region/u_ocl_region/dr_i/gemxKernel_2/inst/kernelOpLow_U0/grp_runGemm_fu_*/grp_GemmBlocks_fu_*/GemmWrite_U0]] -clear_locs
add_cells_to_pblock pblock_CL_mid [get_cells [list CL/xcl_design_i/expanded_region/u_ocl_region/dr_i/gemxKernel_1]] -clear_locs
add_cells_to_pblock pblock_1 [get_cells [list CL/xcl_design_i/expanded_region/u_ocl_region/dr_i/gemxKernel_1/inst/kernelOpLow_U0/grp_runGemm_fu_*/grp_GemmBlocks_fu_*/GemmWrite_U0]] -clear_locs
