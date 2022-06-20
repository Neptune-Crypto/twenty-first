pub enum BaseProcessorTableColumn {
    CLK,
    IP,
    CI,
    NIA,
    IB0,
    IB1,
    IB2,
    IB3,
    IB4,
    IB5,
    JSP,
    JSO,
    JSD,
    ST0,
    ST1,
    ST2,
    ST3,
    ST4,
    ST5,
    ST6,
    ST7,
    ST8,
    ST9,
    ST10,
    ST11,
    ST12,
    ST13,
    ST14,
    ST15,
    INV,
    OSP,
    OSV,
    HV0,
    HV1,
    HV2,
    HV3,
    HV4,
    RAMP,
    RAMV,
}

impl From<BaseProcessorTableColumn> for usize {
    fn from(c: BaseProcessorTableColumn) -> Self {
        match c {
            BaseProcessorTableColumn::CLK => 0,
            BaseProcessorTableColumn::IP => 1,
            BaseProcessorTableColumn::CI => 2,
            BaseProcessorTableColumn::NIA => 3,
            BaseProcessorTableColumn::IB0 => 4,
            BaseProcessorTableColumn::IB1 => 5,
            BaseProcessorTableColumn::IB2 => 6,
            BaseProcessorTableColumn::IB3 => 7,
            BaseProcessorTableColumn::IB4 => 8,
            BaseProcessorTableColumn::IB5 => 9,
            BaseProcessorTableColumn::JSP => 10,
            BaseProcessorTableColumn::JSO => 11,
            BaseProcessorTableColumn::JSD => 12,
            BaseProcessorTableColumn::ST0 => 13,
            BaseProcessorTableColumn::ST1 => 14,
            BaseProcessorTableColumn::ST2 => 15,
            BaseProcessorTableColumn::ST3 => 16,
            BaseProcessorTableColumn::ST4 => 17,
            BaseProcessorTableColumn::ST5 => 18,
            BaseProcessorTableColumn::ST6 => 19,
            BaseProcessorTableColumn::ST7 => 20,
            BaseProcessorTableColumn::ST8 => 21,
            BaseProcessorTableColumn::ST9 => 22,
            BaseProcessorTableColumn::ST10 => 23,
            BaseProcessorTableColumn::ST11 => 24,
            BaseProcessorTableColumn::ST12 => 25,
            BaseProcessorTableColumn::ST13 => 26,
            BaseProcessorTableColumn::ST14 => 27,
            BaseProcessorTableColumn::ST15 => 28,
            BaseProcessorTableColumn::INV => 29,
            BaseProcessorTableColumn::OSP => 30,
            BaseProcessorTableColumn::OSV => 31,
            BaseProcessorTableColumn::HV0 => 32,
            BaseProcessorTableColumn::HV1 => 33,
            BaseProcessorTableColumn::HV2 => 34,
            BaseProcessorTableColumn::HV3 => 35,
            BaseProcessorTableColumn::HV4 => 36,
            BaseProcessorTableColumn::RAMP => 37,
            BaseProcessorTableColumn::RAMV => 38,
        }
    }
}

pub enum IOTableColumn {
    IOSymbol,
}

impl From<IOTableColumn> for usize {
    fn from(c: IOTableColumn) -> Self {
        match c {
            IOTableColumn::IOSymbol => 0,
        }
    }
}

pub enum ProgramTableColumn {
    Address,
    Instruction,
}

impl From<ProgramTableColumn> for usize {
    fn from(c: ProgramTableColumn) -> Self {
        match c {
            ProgramTableColumn::Address => 0,
            ProgramTableColumn::Instruction => 1,
        }
    }
}

pub enum InstructionTableColumn {
    Address,
    CI,
    NIA,
}

impl From<InstructionTableColumn> for usize {
    fn from(c: InstructionTableColumn) -> Self {
        match c {
            InstructionTableColumn::Address => 0,
            InstructionTableColumn::CI => 1,
            InstructionTableColumn::NIA => 2,
        }
    }
}

pub enum OpStackTableColumn {
    CLK,
    CI,
    OSV,
    OSP,
}

impl From<OpStackTableColumn> for usize {
    fn from(c: OpStackTableColumn) -> Self {
        match c {
            OpStackTableColumn::CLK => 0,
            OpStackTableColumn::CI => 1,
            OpStackTableColumn::OSV => 2,
            OpStackTableColumn::OSP => 3,
        }
    }
}

pub enum RAMTableColumn {
    CLK,
    RAMP,
    RAMV,
}

impl From<RAMTableColumn> for usize {
    fn from(c: RAMTableColumn) -> Self {
        match c {
            RAMTableColumn::CLK => 0,
            RAMTableColumn::RAMP => 1,
            RAMTableColumn::RAMV => 2,
        }
    }
}

pub enum JumpStackTableColumn {
    CLK,
    CI,
    JSP,
    JSO,
    JSD,
}

impl From<JumpStackTableColumn> for usize {
    fn from(c: JumpStackTableColumn) -> Self {
        match c {
            JumpStackTableColumn::CLK => 0,
            JumpStackTableColumn::CI => 1,
            JumpStackTableColumn::JSP => 2,
            JumpStackTableColumn::JSO => 3,
            JumpStackTableColumn::JSD => 4,
        }
    }
}

pub enum HashTableColumn {
    RoundNumber,
    AUX0,
    AUX1,
    AUX2,
    AUX3,
    AUX4,
    AUX5,
    AUX6,
    AUX7,
    AUX8,
    AUX9,
    AUX10,
    AUX11,
    AUX12,
    AUX13,
    AUX14,
    AUX15,
}

impl From<HashTableColumn> for usize {
    fn from(c: HashTableColumn) -> Self {
        match c {
            HashTableColumn::RoundNumber => 0,
            HashTableColumn::AUX0 => 1,
            HashTableColumn::AUX1 => 2,
            HashTableColumn::AUX2 => 3,
            HashTableColumn::AUX3 => 4,
            HashTableColumn::AUX4 => 5,
            HashTableColumn::AUX5 => 6,
            HashTableColumn::AUX6 => 7,
            HashTableColumn::AUX7 => 8,
            HashTableColumn::AUX8 => 9,
            HashTableColumn::AUX9 => 10,
            HashTableColumn::AUX10 => 11,
            HashTableColumn::AUX11 => 12,
            HashTableColumn::AUX12 => 13,
            HashTableColumn::AUX13 => 14,
            HashTableColumn::AUX14 => 15,
            HashTableColumn::AUX15 => 16,
        }
    }
}

pub enum U32OpTableColumn {
    IDC,
    LHS,
    RHS,
    LT,
    AND,
    XOR,
    REV,
}

impl From<U32OpTableColumn> for usize {
    fn from(c: U32OpTableColumn) -> Self {
        match c {
            U32OpTableColumn::IDC => 0,
            U32OpTableColumn::LHS => 1,
            U32OpTableColumn::RHS => 2,
            U32OpTableColumn::LT => 3,
            U32OpTableColumn::AND => 4,
            U32OpTableColumn::XOR => 5,
            U32OpTableColumn::REV => 6,
        }
    }
}
