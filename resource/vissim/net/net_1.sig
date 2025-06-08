<?xml version="1.0" encoding="UTF-8"?>
<sc id="1" name=" 珠海市柠溪-兴业固定配时方案16" frequency="1" defaultIntergreenMatrix="0">
  <sgs>
    <sg id="1" name="相位A：北直、北左" defaultSignalSequence="7">
      <defaultDurations>
        <defaultDuration display="4" duration="3000" />
        <defaultDuration display="1" duration="0" />
        <defaultDuration display="3" duration="0" />
      </defaultDurations>
    </sg>
    <sg id="2" name="相位B：南直、南左" defaultSignalSequence="7">
      <defaultDurations>
        <defaultDuration display="4" duration="3000" />
        <defaultDuration display="1" duration="0" />
        <defaultDuration display="3" duration="0" />
      </defaultDurations>
    </sg>
    <sg id="3" name="相位C：东北直、东北左，西南直、新西南左" defaultSignalSequence="7">
      <defaultDurations>
        <defaultDuration display="4" duration="3000" />
        <defaultDuration display="1" duration="0" />
        <defaultDuration display="3" duration="0" />
      </defaultDurations>
    </sg>
  </sgs>
  <intergreenmatrices />
  <progs>
    <prog id="1" cycletime="170000" switchpoint="0" offset="0" intergreens="0" name="信号程序">
      <sgs>
        <sg sg_id="1" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="1000" />
            <cmd display="1" begin="51000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="3000" />
          </fixedstates>
        </sg>
        <sg sg_id="2" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="53000" />
            <cmd display="1" begin="111000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="3000" />
          </fixedstates>
        </sg>
        <sg sg_id="3" signal_sequence="7">
          <cmds>
            <cmd display="3" begin="113000" />
            <cmd display="1" begin="169000" />
          </cmds>
          <fixedstates>
            <fixedstate display="4" duration="3000" />
          </fixedstates>
        </sg>
      </sgs>
    </prog>
  </progs>
  <stages />
  <interstageProgs />
  <stageProgs />
  <dailyProgLists />
</sc>