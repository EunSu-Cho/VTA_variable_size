#ifndef _GEN_INST_AND_UOP_H__
#define _GEN_INST_AND_UOP_H__

#define LSINST_LO(opcode, dept, buf, sram, dram, unused) \
  (((uint64_t)(opcode)  << 0)  | \
   ((uint64_t)(dept)    << 3)  | \
   ((uint64_t)(buf)     << 7)  | \
   ((uint64_t)(sram)    << 10) | \
   ((uint64_t)(dram)    << 26) | \
   ((uint64_t)(unused)  << 58))
#define LSINST_HI(ysz, xsz, xstrd, ypt, ypb, xpl, xpr) \
  (((uint64_t)(ysz)     << 0)  | \
   ((uint64_t)(xsz)     << 16) | \
   ((uint64_t)(xstrd)   << 32) | \
   ((uint64_t)(ypt)     << 48) | \
   ((uint64_t)(ypb)     << 52) | \
   ((uint64_t)(xpl)     << 56) | \
   ((uint64_t)(xpr)     << 60))

#define GMINST_LO(opcode, dept, rst, uopbgn, uopend, lpext0, lpext1, unused) \
  (((uint64_t)(opcode)  << 0)  | \
   ((uint64_t)(dept)    << 3)  | \
   ((uint64_t)(rst)     << 7)  | \
   ((uint64_t)(uopbgn)  << 8)  | \
   ((uint64_t)(uopend)  << 21) | \
   ((uint64_t)(lpext0)  << 35) | \
   ((uint64_t)(lpext1)  << 49) | \
   ((uint64_t)(unused)  << 63))
#define GMINST_HI(accidx0, accidx1, inpidx0, inpidx1, wgtidx0, wgtidx1) \
  (((uint64_t)(accidx0) << 0)  | \
   ((uint64_t)(accidx1) << 11) | \
   ((uint64_t)(inpidx0) << 22) | \
   ((uint64_t)(inpidx1) << 33) | \
   ((uint64_t)(wgtidx0) << 44) | \
   ((uint64_t)(wgtidx1) << 54))

#define FNINST_LO(opcode, dept, unused) \
  (((uint64_t)(opcode)  << 0)  | \
   ((uint64_t)(dept)    << 3)  | \
   ((uint64_t)(unused)  << 7))
#define FNINST_HI(unused) \
  (((uint64_t)(unused)  << 0))

#define UOP(accidx, inpidx, wgtidx) \
  (((uint32_t)(accidx) << 0)  | \
   ((uint32_t)(inpidx) << 11) | \
   ((uint32_t)(wgtidx) << 22))
#endif