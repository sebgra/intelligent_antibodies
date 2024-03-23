<template>
  <div class="table">
    <div class="table_header">
      <div class="table_sequence"><b class="weight">Sequence</b></div>
      <div class="table_score"><b class="weight">Score</b></div>
      <div class="table_button"><b class="weight">Action</b></div>
    </div>
    <div :key="line[0]" class="table_line" v-for="line in datas">
      <div class="table_sequence">
        {{ line[0] }}
      </div>
      <div class="table_score">
        {{ line[1] }}
      </div>
      <div class="table_button">
        <button>View</button>
      </div>
    </div>
  </div>
</template>
<script>
export default {
  name:'Table',
  data() {
    return {
      novar:42

    }
  },
  props: {
    datas: Array
  }
}
</script>


<style scoped>
.table {
  display: block;
  height:fit-content;
  border: 1px solid lightgrey;
  border-radius: 5px;
  justify-content: left;
}
.table_header {
  display: flex;
  justify-content: space-between;
  background-color: #409d9e;
  color: white;
  border-radius: 5px 5px 0 0;
}
.weight {
  font-weight: 700;
}
.table_line {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid lightgrey;
}
.table_line:last-child{
  border-bottom: none
}
.table_line:hover {
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid lightgrey;
  background-color: lightgrey
}
.table_sequence {
  width: 75%;
  border-right: 1px solid lightgrey;
  padding: 5px;
}
.table_score {
  width: 15%;
  padding: 5px;
  border-right: 1px solid lightgrey;
}
.table_button {
  width: 2rem;
  padding: 5px;
  margin:auto;
}
</style>
