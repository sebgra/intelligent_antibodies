<template>
  <div>
    <div class='prot' id="pvpanel" ref="pvpanel"></div>
  </div>
</template>

<script>
import pv from "bio-pv"
export default {
  name: "ProteinViewPanel",
  props: {
    protId: {
      type:String
    }
  },
  mounted() {
    this.$nextTick(
      () => {
        let viewer = pv.Viewer(this.$refs.pvpanel, {
          width: 600,
          height: 600,
          antialias: true,
          quality: "medium",
          background: "black"
        });
        let prot = "/src/components/_static/" + this.protId +".pdb"
        viewer.on("viewerReady", function() {
          pv.io.fetchPdb(prot, function(structure) {
              viewer.cartoon('mol', structure);
              viewer.fitTo(structure)
          });
        });
      }
    )
  },
}
</script>
<style scoped>
.prot {
  display: flex;
  justify-content: center;
  padding: 10px
}
</style>