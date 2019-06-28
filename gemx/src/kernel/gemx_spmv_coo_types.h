/**********
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * **********/
/**
 *  @brief GEMX SPMV COO Format datatypes for HLS kernel code.
 *
 */

#ifndef GEMX_SPMV_COO_TYPES_H
#define GEMX_SPMV_COO_TYPES_H

#include "gemx_types.h"


namespace gemx {

template <typename t_FloatType, typename t_IdxType, unsigned int t_NumBanks, unsigned int t_NumValsPerUnit>
class SpmCoo {
	private:
		t_FloatType m_Val;
		t_IdxType m_Col;
		t_IdxType m_Row;
	public:
		SpmCoo(){}
		SpmCoo(t_FloatType p_val, t_IdxType p_row, t_IdxType p_col)
			: m_Val(p_val), m_Row(p_row), m_Col(p_col) {}
		t_IdxType &getCol() {return m_Col;}
		t_IdxType &getRow() {return m_Row;}
		t_FloatType &getVal() {return m_Val;}
		unsigned int getColBank() {return CalcMod<unsigned int>(m_Col, t_NumBanks);}
    unsigned int getColOffset() {return m_Col / (t_NumBanks * t_NumValsPerUnit) ;}
    unsigned int getColIndex() {return CalcMod<unsigned int>(m_Col, t_NumValsPerUnit, t_NumBanks);} 
		void
    print(std::ostream& os) {
        os << std::setw(GEMX_FLOAT_WIDTH) << getRow() << " "
           << std::setw(GEMX_FLOAT_WIDTH) << getCol() << "    "
           << std::setw(GEMX_FLOAT_WIDTH) << getVal();
      }
};

template <typename T1, typename T2,  unsigned int T3, unsigned int T4>
std::ostream& operator<<(std::ostream& os, SpmCoo<T1, T2, T3, T4>& p_val) {
  p_val.print(os);
  return(os);
}

template <typename t_FloatType, typename t_IdxType, unsigned int t_NumBanks, unsigned int t_NumValsPerUnit>
class SpmCol {
	private:
		t_FloatType m_Val;
		t_IdxType m_Row;
	public:
		SpmCol(){}
		SpmCol(t_FloatType p_val, t_IdxType p_row)
			: m_Val(p_val), m_Row(p_row) {}
		t_IdxType &getRow() {
		#pragma HLS inline self
			return m_Row;
		}
		t_FloatType &getVal() {
		#pragma HLS inline self
			return m_Val;
		}
		unsigned int getRowBank() {
		#pragma HLS inline self
			return CalcMod<unsigned int>(m_Row, t_NumBanks);
		}
    unsigned int getRowOffset() {
		#pragma HLS inline self
			return m_Row / (t_NumBanks * t_NumValsPerUnit);
		}
    unsigned int getRowIndex() {
		#pragma HLS inline self
			return CalcMod<unsigned int>(m_Row, t_NumValsPerUnit, t_NumBanks);
		} 
		void
    print(std::ostream& os) {
        os << std::setw(GEMX_FLOAT_WIDTH) << getRow() << " "
           << std::setw(GEMX_FLOAT_WIDTH) << getVal();
      }
};

template <typename T1, typename T2, unsigned int T3, unsigned int T4>
std::ostream& operator<<(std::ostream& os, SpmCol<T1, T2, T3, T4>& p_val) {
	p_val.print(os);
	return(os);
}

template <typename t_FloatType, typename t_IdxType, unsigned int t_NumBanks, unsigned int t_NumGroups, unsigned int t_NumValsPerUnit, unsigned int t_Width>
class SpmSameCol {
	public:
		typedef SpmCol<t_FloatType, t_IdxType, t_NumBanks, t_NumValsPerUnit*t_NumGroups> SpmColType;
		typedef SpmCoo<t_FloatType, t_IdxType, t_NumBanks, t_NumValsPerUnit> SpmCooType;
		typedef WideType<SpmCooType, t_Width> SpmCooWideType;

	private:
		t_IdxType m_Col;
		SpmColType m_SpmCols[t_Width];
	public:
		t_IdxType &getCol(){return m_Col;}
		SpmColType &operator[](unsigned int p_idx) {return m_SpmCols[p_idx];}
		SpmSameCol() {}
		void 
		init (t_IdxType p_col, SpmCooWideType p_coo) {
			m_Col = p_col;
			for (int i = 0; i < t_Width; ++i) {
				#pragma HLS UNROLL
				m_SpmCols[i].getRow() = p_coo[i].getRow();
				m_SpmCols[i].getVal() = p_coo[i].getVal();	
			}
		}
		unsigned int getColBank() {return CalcMod<unsigned int>(m_Col, t_NumBanks);}
    unsigned int getColOffset() {return m_Col / (t_NumBanks * t_NumValsPerUnit) ;}
    unsigned int getColIndex() {return CalcMod<unsigned int>(m_Col, t_NumValsPerUnit, t_NumBanks);} 
		void
    print(std::ostream& os) {
    	os << std::setw(GEMX_FLOAT_WIDTH) << getCol() << " ";
			for (int i = 0; i < t_Width; ++i) {
				os << m_SpmCols[i];
			}
    }
};

template <typename T1, typename T2, unsigned int T3, unsigned int T4, unsigned int T5, unsigned int T6>
std::ostream& operator<<(std::ostream& os, SpmSameCol<T1, T2, T3, T4, T5, T6>& p_val) {
	p_val.print(os);
	return(os);
}

template <typename t_FloatType, typename t_IdxType, unsigned int t_NumBanks, unsigned int t_NumValsPerUnit>
class SpmAB {
	private:
		t_FloatType m_ValA;
		t_FloatType m_ValB;
		t_IdxType m_Row;

	public:
		SpmAB() {}
		SpmAB(t_FloatType p_A, t_FloatType p_B, unsigned int p_row)
			:m_ValA(p_A), m_ValB(p_B), m_Row(p_row){}

		t_FloatType getA(){return m_ValA;}
		t_FloatType getB(){return m_ValB;}
		t_IdxType getRow(){return m_Row;}
		unsigned int getRowBank() {return CalcMod<unsigned int>(m_Row, t_NumBanks);}
    unsigned int getRowOffset() {return m_Row / (t_NumBanks * t_NumValsPerUnit);}
    unsigned int getRowIndex() {return CalcMod<unsigned int>(m_Row, t_NumValsPerUnit, t_NumBanks);} 
		void
		print(std::ostream& os) {
			 os << std::setw(GEMX_FLOAT_WIDTH) << getRow() << " "
					<< std::setw(GEMX_FLOAT_WIDTH) << getB() << " "
					<< std::setw(GEMX_FLOAT_WIDTH) << getA();
		}
};

template <typename T1, typename T2, unsigned int T3, unsigned int T4>
std::ostream& operator<<(std::ostream& os, SpmAB<T1, T2, T3, T4>& p_val) {
	p_val.print(os);
	return(os);
}

template <typename t_FloatType, typename t_IdxType, unsigned int t_NumBanks, unsigned int t_NumValsPerUnit>
class SpmC {
	private:
		unsigned int m_RowOffset;

	public:
		t_FloatType m_Val[t_NumValsPerUnit];
		SpmC() {}
		void init (t_FloatType p_data, unsigned int p_rowOffset) {
		#pragma HLS inline self
			for (unsigned int i=0; i<t_NumValsPerUnit; ++i) {
			#pragma HLS UNROLL
				m_Val[i] = p_data;
			}
			m_RowOffset=p_rowOffset;
		}

		t_FloatType &getVal(unsigned int i){
		#pragma HLS inline self
			return m_Val[i];
		}
		t_FloatType &operator[](unsigned int i){
		#pragma HLS inline self
		#pragma HLS ARRAY_PARTITION variable=m_Val complete
			return m_Val[i];
		}
		unsigned int &getRowOffset(){
		#pragma HLS inline self
			return m_RowOffset;
		}
		void
		print(std::ostream& os) {
			for (int i=0; i < t_NumValsPerUnit; ++i) {
				os << std::setw(GEMX_FLOAT_WIDTH) << getVal(i) << " ";
				os << getRowOffset() << " ";
			}
		}
};

template <typename T1, typename T2, unsigned int T3, unsigned int T4>
std::ostream& operator<<(std::ostream& os, SpmC<T1, T2, T3, T4>& p_val) {
	p_val.print(os);
	return(os);
}

} // namespace
#endif
